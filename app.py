import streamlit as st
import json
from datetime import datetime
import os
import uuid
import time
import streamlit as st
import requests


st.set_page_config(page_title="Chat Assistant", layout="centered")
INGREDIENTS_FILE = "ingredients.json"
DATA_FILE = "chats.json"
BACKEND_URL = "http://localhost:8000"  # http://backend:8000

def load_ingredients():
    if os.path.exists(INGREDIENTS_FILE):
        with open(INGREDIENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_ingredients(data):
    with open(INGREDIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chats():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)

def generate_chat_name(text):
    t = text.strip().split("\n")[0][:40]
    return f"{t}..." if len(t) > 40 else t

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.chats = load_chats()
    st.session_state.current_chat_id = None  # start always from empty chat on full reload

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Chat", "Add Ingredients", "Feedback"])

if page == "Chat":
    st.sidebar.markdown("### Chats")
    if st.sidebar.button("Start new chat"):
        st.session_state.current_chat_id = None

    # sorting chats by creation date - the top once are the most recent ones
    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )

    for cid, chat in sorted_chats:
        if st.sidebar.button(chat["name"], key=cid):
            st.session_state.current_chat_id = cid

if page == "Home":
    st.title("Food Waste Manager Assistant")
    st.subheader("Welcome")
    st.markdown(
        """
        This assistant helps users reduce food waste and plan meals.

        You can ask:
        - what meals you can cook with the ingredients you have  
        - which products should be used first based on expiration dates  
        - to filter recipes by price, cooking time, dietary restrictions or ratings  

        The assistant can also extract recipes from blogs or images and suggest menu options for complex scenarios.
        """
    )

elif page == "Chat":
    st.title("Food Waste Manager Assistant")

    if st.session_state.current_chat_id:
        chat = st.session_state.chats[st.session_state.current_chat_id]
        for msg in chat["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Type your message"):
            chat["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            payload = {
                "message": prompt
            }

            try:
                r = requests.post(
                    f"{BACKEND_URL}/agent/message",
                    data={
                        "message": prompt,
                        "user_id": st.session_state.current_chat_id
                    },
                    timeout=30
                )
                r.raise_for_status()
                response = r.json().get("message", "No response from backend")
            except Exception as e:
                response = f"Backend error: {e}"

            chat["messages"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            save_chats(st.session_state.chats)

    else:
        if prompt := st.chat_input("Type your message"):
            new_id = str(uuid.uuid4())
            name = generate_chat_name(prompt)
            payload = {
                "message": prompt
            }

            try:
                r = requests.post(
                    f"{BACKEND_URL}/agent/message",
                    data={
                        "message": prompt,
                        "user_id": st.session_state.current_chat_id
                    },
                    timeout=30
                )
                r.raise_for_status()
                response = r.json().get("message", "No response from backend")
            except Exception as e:
                response = f"Backend error: {e}"

            st.session_state.chats[new_id] = {
                "name": name,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "created_at": datetime.now().isoformat()
            }
            st.session_state.current_chat_id = new_id
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            save_chats(st.session_state.chats)

elif page == "Add Ingredients":
    st.title("Add Ingredient")

    if "ingredient_name" not in st.session_state:
        st.session_state.ingredient_name = ""
    if "ingredient_quantity" not in st.session_state:
        st.session_state.ingredient_quantity = 0
    if "ingredient_unit" not in st.session_state:
        st.session_state.ingredient_unit = "g"
    if "ingredient_category" not in st.session_state:
        st.session_state.ingredient_category = "Vegetables"

    st.markdown("Fill in the details of the ingredient you want to add.")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Product Name", value=st.session_state.ingredient_name)

        category = st.selectbox(
            "Category",
            [
                "Vegetables",
                "Fruits",
                "Dairy",
                "Meat",
                "Fish",
                "Grains",
                "Spices",
                "Beverages",
                "Snacks",
                "Other"
            ],
            index=0
        )

    with col2:
        expiration = st.date_input("Expiration Date")

        quantity = st.number_input("Quantity", min_value=0.0, step=1.0)

        unit = st.selectbox("Unit", ["g", "ml", "pcs"])

    if st.button("Add Ingredient"):
        name = name.strip()

        if not name:
            st.warning("Please provide a product name.")
        else:
            entry = {
                "id": str(uuid.uuid4()),
                "name": name,
                "category": category,
                "expiration_date": expiration.isoformat(),
                "quantity": quantity,
                "unit": unit,
                "created_at": datetime.now().isoformat()
            }

            ingredients = load_ingredients()
            ingredients.append(entry)
            save_ingredients(ingredients)

            st.success(f"Ingredient '{name}' added successfully.")

            # Clear fields
            st.session_state.ingredient_name = ""
            st.session_state.ingredient_quantity = 0
            st.session_state.ingredient_unit = "g"
            st.session_state.ingredient_category = "Vegetables"

            time.sleep(1)
            st.rerun()

elif page == "Feedback":
    st.title("Feedback")
    if "feedback_name" not in st.session_state:
        st.session_state.feedback_name = ""
    if "feedback_comment" not in st.session_state:
        st.session_state.feedback_comment = ""

    name = st.text_input("Name", value=st.session_state.feedback_name, key="feedback_name_input")
    comment = st.text_area("Feedback", value=st.session_state.feedback_comment, key="feedback_comment_input")
    rating = st.slider("Rating", 1, 5, 3)

    if st.button("Submit"):
        name = name.strip()
        comment = comment.strip()

        if not name and not comment:
            st.warning("Please provide your name and feedback before submitting.")
        elif not name:
            st.warning("Please provide your name before submitting.")
        elif not comment:
            st.warning("Please provide feedback before submitting.")
        else:
            st.success("Thank you for your feedback!")
            time.sleep(2)
            st.session_state.feedback_name = ""
            st.session_state.feedback_comment = ""
            time.sleep(2)
            st.rerun()

st.markdown("---")
st.caption(f"© {datetime.now().year} Food Waste Manager Assistant")