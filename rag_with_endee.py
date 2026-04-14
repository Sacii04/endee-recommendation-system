

try:
    from endee import Endee
except:
    print("Endee not installed, running in simulation mode")
    Endee = None


def load_data():
    with open("data.txt", "r") as f:
        return f.read().split(".")


def main():
    print("RAG Chatbot with Endee 🤖")

    data = load_data()

    # initialize Endee if available
    db = Endee() if Endee else None

    # store data
    if db:
        for text in data:
            db.add(text)

    while True:
        query = input("\nAsk something (type exit): ").lower()

        if query == "exit":
            break

        if db:
            results = db.search(query)
            print("Results from Endee:")
            for r in results:
                print("-", r)
        else:
            # fallback (your current logic)
            print("Simulation mode (no Endee):")
            for line in data:
                if query in line.lower():
                    print("-", line)


if __name__ == "__main__":
    main()
