from re import L
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Union, TypedDict, Annotated, Tuple
import operator
from langgraph.graph import END
from langchain.agents import create_react_agent
from tools import web_search, extract_article, summarize_text, write_to_file, generate_insta_post, send_email
from langchain_core.tools import Tool
from models import Response, Act, Plan, PlanExecute





tools = [
        Tool(
            name="web_search",
            func=web_search,
            description="Useful for when you need to answer questions about current events."
        ),
        Tool(
            name="extract_article",
            func=extract_article,
            description="Useful for when you need to extract the title and text from an article."
        ),
        Tool(
            name="summarize_text",
            func=summarize_text,
            description="Useful for when you need to summarize a long text."
        ),
        Tool(
            name="write_to_file",
            func=write_to_file,
            description="Useful for when you need to write text to a file."
        ),
        Tool(
            name="generate_insta_post",
            func=generate_insta_post,
            description="Useful for when you need to generate a instagram post using a chain and input data."
        ),
        Tool(
            name="send_email",
            func=send_email,
            description="Useful for when you need to send an email."
        )
    ]

custom_template = """You are a helpful assistant.

    You have access to the following tools:
    {tools}

    You can use these tools: {tool_names}

    You must always respond in one of the following formats:

    To use a tool:
    Thought: your reasoning here
    Action: name of the tool
    Action Input: input to the tool

    To return a final answer:
    Thought: your reasoning here
    Final Answer: the final answer to the user's question

    !Do not respond with any other text other than the above formats!

    !The format should be strictly followed!

    Here is the user input: 
    {input}

    Begin!

    Use the scratchpad below to plan your actions step by step:
    {agent_scratchpad}
"""
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
prompt = PromptTemplate(template=custom_template, input_variables=["input", "tools", "tool_names", "agent_scratchpad"])

agent_executor = create_react_agent(llm, tools, prompt=prompt)

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            The steps should be given in the format:
            Action 1: <action>
            Action 2: <action>
            ...
            """,
        ),
        (
            "placeholder", 
            "{messages}"
        ),
    ]
)




replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective was this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the following steps:
    {past_steps}

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)
# config = RunnableConfig(metadata=True)

planner = planner_prompt | llm.with_structured_output(Plan)
replanner = replanner_prompt | llm.with_structured_output(Act)


def plan_step(state: PlanExecute):
    global total_tokens
    print('\n Planning...')
    plan = planner.invoke({
            "messages": [
                ("user", state.input),
            ]
        })
    state.plan = plan.steps
    #print("\nSteps: ", plan.steps)
    return state



def execute_step(state: PlanExecute):
    global total_tokens
    print('\n Executing step...')
    plan = state.plan
    
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""

    # formatted_prompt = prompt.format(
    # input=task_formatted,
    # tools="\n".join([f"{t.name}: {t.description}" for t in tools]),
    # tool_names=", ".join([t.name for t in tools]),
    # agent_scratchpad=""
    # )

    agent_response = agent_executor.invoke({
        "input": task_formatted,
        "intermediate_steps": []  # required by ReAct agent
    })
    print('\n Agent result start: \n', agent_response, '\nAgent Execution end: \n')
    #print('appended: ', agent_response.messages[-1].content)
    state.past_steps.append((task, agent_response.messages[-1].content))
    return state

def replan_step(state: PlanExecute):
    global total_tokens
    print('\n Replanning...')
    act = replanner.invoke({
        "input": state.input,
        "plan": state.plan,
        "past_steps": state.past_steps
    })
    if isinstance(act.action, Response):
        state.response = act.action.response
        return state
    else:
        state.plan = act.action.steps
        return state

def should_end(state: PlanExecute):
    if "response" in state and state.response:
        print('\n Response: ', state.response)
        return END
    else:
        print('\n Agent: ', state.past_steps[-1][1])
        return "agent"


# user_input = "Create an instagram post for the latest news in Automobile industry"
# state = PlanExecute(input=user_input)
# state = plan_step(state)
# while True:
#     state = execute_step(state)
#     if should_end(state) == END:
#         break
        
#     state = replan_step(state)


# from langgraph.graph import StateGraph, START

# workflow = StateGraph(PlanExecute)

# # Add the plan node
# workflow.add_node("planner", plan_step)

# # Add the execution step
# workflow.add_node("agent", execute_step)

# # Add a replan node
# workflow.add_node("replan", replan_step)

# workflow.add_edge(START, "planner")

# # From plan we go to agent
# workflow.add_edge("planner", "agent")

# # From agent, we replan
# workflow.add_edge("agent", "replan")

# workflow.add_conditional_edges(
#     "replan",
#     # Next, we pass in the function that will determine which node is called next.
#     should_end,
#     ["agent", END],
# )

# # Finally, we compile it!
# # This compiles it into a LangChain Runnable,
# # meaning you can use it as you would any other runnable
# app = workflow.compile()

# config = {"recursion_limit": 50}
# inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
# for event in app.astream(inputs, config=config):
#     for k, v in event.items():
#         if k != "__end__":
#             #print(v)


