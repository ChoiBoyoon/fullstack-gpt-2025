from langchain_openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.example_selector import LengthBasedExampleSelector

chat = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

examples = [
    {"movie":"Avatar", 
     "info":"Title: Avatar(2009)\nDirector: James Cameron\nMain Cast: Sam Worthington, Zoe Saldaña, Sigourney Weaver\nBudget: $237 million\nBox Office: $2.92 billion\nGenre: Science Fiction, Action, Adventure\nSynopsis: Set on the lush alien world of Pandora, a paraplegic marine is dispatched to infiltrate the native Na'vi tribe, but becomes torn between following his orders and protecting the world he feels is his home."},
    {"movie":"Avengers Endgame", 
    "info":"Title: Avengers: Endgame(2019)\nDirector: Anthony and Joe Russo\nMain Cast: Robert Downey Jr., Chris Evans, Scarlett Johansson\nBudget: $356 million\nBox Office: $2.80 billion\nGenre: Superhero, Action, Sci-Fi\nSynopsis: The Avengers assemble once more to undo the catastrophic events caused by Thanos in their previous battle, leading to a final showdown to save the universe."},
    {"movie":"Avatar The Way of Water", 
    "info":"Title: Avatar: The Way of Water (2022)\nDirector: James Cameron\nMain Cast: Sam Worthington, Zoe Saldaña, Sigourney Weaver\nBudget: $350 million\nBox Office: $2.32 billion\nGenre: Science Fiction, Action, Adventure\nSynopsis: Jake and Neytiri have formed a family, but when an old threat resurfaces, they are forced to leave their home and explore the regions of Pandora, including the oceans."},
    {"movie":"Titanic", 
    "info":"Title: Titanic (1997)\nDirector: James Cameron\nMain Cast: Leonardo DiCaprio, Kate Winslet, Billy Zane\nBudget: $200 million\nBox Office: $2.23 billion\nGenre: Romance, Drama\nSynopsis: A young couple from different social backgrounds fall in love aboard the ill-fated R.M.S. Titanic, leading to a tragic and unforgettable love story."},
    {"movie":"Star Wars: The Force Awakens",
     "info":"Title: Star Wars: The Force Awakens (2015)\nDirector: J.J. Abrams\nMain Cast: Daisy Ridley, John Boyega, Harrison Ford\nBudget: $245 million\nBox Office: $2.06 billion\nGenre: Science Fiction, Action, Adventure\nSynopsis: A new heroine emerges in the galaxy as the Resistance faces the First Order, leading to the return of familiar faces and the discovery of new heroes."}
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Tell me about {movie}."),
    ("ai","{info}")
])

example_selector = LengthBasedExampleSelector(
    examples = examples,
    example_prompt = example_prompt,
    max_length=1000 #최대 1000토큰
)

fewshot_example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    example_selector = example_selector,
    suffix="Human: Tell me about {movie}.",
    input_variables=["movie"]
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a movie expert"),
    fewshot_example_prompt, #length-base로 선택된 예시들
    ("human", "What do you know about {movie}?")
])

final_prompt.format(movie="Millenium")

# chain = final_prompt | chat
# chain.invoke("Millenium")
