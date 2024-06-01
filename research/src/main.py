from dotenv import load_dotenv
from crewai import Crew
from tasks import MeetingPrepTasks
from agents import MeetingPrepAgents
from langchain_google_genai import ChatGoogleGenerativeAI





def main():
    load_dotenv()
    GEMINI_API_KEY = "AIzaSyAvvRcgQ6jwYTm49JVEgYgZ23gWE4-g54w"

    print("## Welcome to the Meeting Prep Crew")
    print('-------------------------------')
    meeting_participants = "narendra@gmail.com karthik@gmail.com ,deepak@gmail.com"
    meeting_context = "About AI whether it will good or bad"
    meeting_objective = "Finalized expression whether it was helpful or not"

    tasks = MeetingPrepTasks()
    agents = MeetingPrepAgents()
    
    # create agents
    research_agent = agents.research_agent()
    industry_analysis_agent = agents.industry_analysis_agent()
    meeting_strategy_agent = agents.meeting_strategy_agent()
    summary_and_briefing_agent = agents.summary_and_briefing_agent()
    
    # create tasks
    research_task = tasks.research_task(research_agent, meeting_participants, meeting_context)
    industry_analysis_task = tasks.industry_analysis_task(industry_analysis_agent, meeting_context)
    meeting_strategy_task = tasks.meeting_strategy_task(meeting_strategy_agent, meeting_context, meeting_objective)
    summary_and_briefing_task = tasks.summary_and_briefing_task(summary_and_briefing_agent, meeting_context, meeting_objective)
    
    meeting_strategy_task.context = [research_task, industry_analysis_task]
    summary_and_briefing_task.context = [research_task, industry_analysis_task, meeting_strategy_task]
    
    crew = Crew(
      agents=[
        research_agent,
        industry_analysis_agent,
        meeting_strategy_agent,
        summary_and_briefing_agent
      ],
      tasks=[
        research_task,
        industry_analysis_task,
        meeting_strategy_task,
        summary_and_briefing_task
      ],
      verbose=True,
      llm=ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY),

    )
    
    result = crew.kickoff()
    
    print(result)
    
if __name__ == "__main__":
    main()