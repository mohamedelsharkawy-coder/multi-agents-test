import streamlit as st
import os
from pydantic import BaseModel, Field
from typing import List
from crewai import Crew, Agent, Task, Process, LLM
from crewai.tools import tool
from tavily import TavilyClient
from scrapegraph_py import Client
import zipfile
from io import BytesIO

# output folder path 
output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)

# environment
os.environ["GEMINI_API_KEY"] = "AIzaSyAkEJ5NwUtlaEDNwpNXl4iwzfjuyQR4QIw"

# general setup for the llm model
basic_llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0,
    provider="google_ai_studio",
    api_key=os.environ["GEMINI_API_KEY"]
)

# Agent 1
the_max_queries = 20

class search_recommendation(BaseModel):
    search_queries: List[str] = Field(..., title="Recommended searches to be sent to the search engines", min_items=1, max_items=the_max_queries)

search_recommendation_agent = Agent(
    role="search_recommendation_agent",
    goal="""to provide a list of recommendations search queries to be passed to the search engine.
    The queries must be varied and looking for specific items""",
    backstory="The agent is designed to help in looking for products by providing a list of suggested search queries to be passed to the search engine based on the context provided.",
    llm=basic_llm,
    verbose=True,
)

search_recommendation_task = Task(
    description="\n".join([
        "Mariam is looking for a job as {job_name}",
        "so the job must be suitable for {level}",
        "The search query must take the best offers",
        "I need links of the jobs",
        "The recommended query must not be more than {the_max_queries}",
        "The job must be in {country_name}"
    ]),
    expected_output="A JSON object containing a list of suggested search queries.",
    output_json=search_recommendation,
    agent=search_recommendation_agent,
    output_file=os.path.join(output_dir, "step_1_Recommend_search_queries.json"),
)

# Agent 2
search_clint = TavilyClient(api_key="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")

class SingleSearchResult(BaseModel):
    title: str
    url: str = Field(..., title="the page url")
    content: str
    score: float
    search_query: str

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult]

@tool
def search_engine_tool(query: str):
    """Useful for search-based queries. Use this to find current information about any query related pages using a search engine"""
    return search_clint.search(query)

search_engine_agent = Agent(
    role="search engine agent",
    goal="To search on job based on suggested search queries",
    backstory="that agent designed to help in finding jobs by using the suggested search queries",
    llm=basic_llm,
    verbose=True,
    tools=[search_engine_tool]
)

search_engine_task = Task(
    description="\n".join([
        "search for jobs based on the suggested search queries",
        "you have to collect results from the suggested search queries",
        "ignore any results that are not related to the job",
        "Ignore any search results with confidence score less than ({score_th})",
        "the search result will be used to summarize the posts to understand what the candidate needs to have",
        "you should give me more than 10 jobs"
    ]),
    expected_output="A JSON object containing search results.",
    output_json=AllSearchResults,
    agent=search_engine_agent,
    output_file=os.path.join(output_dir, "step_2_search_results.json")
)

# Agent 3
scrape_client = Client(api_key="sgai-320af611-4b53-4425-996f-f8f609c9349f")

class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str

class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title="The original url of the job page")
    Job_Requirements: str = Field(..., title="The requirements of the job")
    Job_Title: str = Field(..., title="The title of the job")
    Job_Details: str = Field(title="The Details of the job", default=None)
    Job_Description: str = Field(..., title="The Description of the job")
    Job_Location: str = Field(title="The location of the job", default=None)
    Job_Salary: str = Field(title="The salary of the job", default=None)
    Job_responsability: str = Field(..., title="The responsibility of the job")
    Job_type: str = Field(title="The type of the job", default=None)
    Job_Overview: str = Field(..., title="The overview of the job")
    qualifications: str = Field(..., title="The qualifications of the job")
    product_specs: List[ProductSpec] = Field(..., title="The specifications of the product. Focus on the most important requirements.", min_items=1, max_items=5)
    agent_recommendation_notes: List[str] = Field(..., title="A set of notes why would you recommend or not recommend this job to the candidate, compared to other jobs.")

class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

@tool
def web_scraping_tool(page_url: str):
    """An AI Tool to help an agent to scrape a web page"""
    details = scrape_client.smartscraper(
        website_url=page_url,
        user_prompt="Extract ```json\n" + SingleExtractedProduct.schema_json() + "```\n From the web page"
    )
    return {
        "page_url": page_url,
        "details": details
    }

search_scrap_agent = Agent(
    role="Web scrap agent to extract url information",
    goal="to extract information from any website",
    backstory="the agent designed to extract required information from any website and that information will be used to understand which skills the jobs need",
    llm=basic_llm,
    verbose=True,
    tools=[web_scraping_tool]
)

search_scrap_task = Task(
    description="\n".join([
        "The task is to extract job details from any job offer page url.",
        "The task has to collect results from multiple pages urls.",
        "you should focus on what requirements or qualification or responsibilities",
        "the results from you the user will use it to understand which skills he needs to have",
        "I need you to give me more than +5 jobs"
    ]),
    expected_output="A JSON object containing jobs details",
    output_json=AllExtractedProducts,
    output_file=os.path.join(output_dir, "step_3_search_results.json"),
    agent=search_scrap_agent
)

# Agent 4
search_summarize_agent = Agent(
    role="extract information about what requirements for every job",
    goal="to extract information about what requirements for every job",
    backstory="the agent should detect what requirements for the job according to the job description and requirements",
    llm=basic_llm,
    verbose=True,
)

search_summarize_task = Task(
    description="\n".join([
        "extract what skills should the candidate of that job should have",
        "you have to collect results about what each job skills need",
        "ignore any results that have None values",
        "Ignore any search results with confidence score less than ({score_th})",
        "the candidate needs to understand what skills he should have",
        "you can also recommend skills from understanding jobs title even if it is not in the job description",
        "I need you to give me +10 skills"
    ]),
    expected_output="Summarize of what skills that job need candidate to have",
    agent=search_summarize_agent,
    output_file=os.path.join(output_dir, "step_4_search_results.json")
)

Company_Crew = Crew(
    process=Process.sequential,
    agents=[search_recommendation_agent, search_engine_agent, search_scrap_agent, search_summarize_agent],
    tasks=[search_recommendation_task, search_engine_task, search_scrap_task, search_summarize_task]
)
######################## Streamlit UI #######################

st.title("Job Search AI Crew")

choice_job = st.text_input("Enter the job title (e.g., AI developer):")
choice_level = st.text_input("Enter the level (e.g., Junior):")
choice_region = st.text_input("Enter the region (e.g., Egypt):")

if "crew_ran" not in st.session_state:
    st.session_state.crew_ran = False

if st.button("Run Crew"):
    if choice_job and choice_level and choice_region:
        with st.spinner("Crew is running, please wait... ⏳"):
            crew_results = Company_Crew.kickoff(
                inputs={
                    "job_name": f"job offer for {choice_job}",
                    "the_max_queries": 20,
                    "level": choice_level,
                    "score_th": 0.02,
                    "country_name": choice_region
                }
            )
        st.success("Crew finished running!")
        st.write("Needed Skills:")
        st.write(crew_results)
        st.session_state.crew_ran = True
    else:
        st.warning("Please enter job title, level, and region.")

if st.session_state.crew_ran:
    files_exist = all(
        os.path.exists(os.path.join(output_dir, fname)) for fname in [
            "step_1_Recommend_search_queries.json",
            "step_2_search_results.json",
            "step_3_search_results.json",
            "step_4_search_results.json"
        ]
    )
    if files_exist:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for filename in [
                "step_1_Recommend_search_queries.json",
                "step_2_search_results.json",
                "step_3_search_results.json",
                "step_4_search_results.json"
            ]:
                file_path = os.path.join(output_dir, filename)
                zip_file.write(file_path, arcname=filename)
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All JSON Files",
            data=zip_buffer,
            file_name="all_job_results.zip",
            mime="application/zip"
        )
    else:
        st.info("Crew ran but no result files found yet.")
