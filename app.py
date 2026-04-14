from recommender import hybrid_recommend
from openai import OpenAI

client = OpenAI()

def explain(query, results):
    context = "\n".join(results['description'].tolist())
    prompt = f"User query: {query}\nRecommendations:\n{context}\nExplain why they match."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

query = input("Enter query: ")
user_id = input("User ID (optional): ")
user_id = int(user_id) if user_id else None

results = hybrid_recommend(query, user_id)
print(results[['title']])
print(explain(query, results))
