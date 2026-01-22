import streamlit as st
import json
from datetime import datetime
import os
import uuid
import time
import requests
import os

DEFAULT_USER_ID = "55bed824-3a3d-48ac-98f2-ec2e284b1e24"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Chat Assistant", layout="centered")
INGREDIENTS_FILE = "ingredients.json"
DATA_FILE = "chats.json"

if "pdf_uploader_key" not in st.session_state:
    st.session_state.pdf_uploader_key = 0

if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

def sanitize_message(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

def render_recipes(recipes):
    for r in recipes:
        with st.container(border=True):
            st.subheader(r.get("title", "Untitled recipe"))
            st.write(f"Total time: {r.get('total_time_minutes', 'N/A')} minutes")

            tags = r.get("tags", [])
            if tags:
                st.caption(", ".join(tags))

            matched = r.get("matched_ingredients", [])
            if matched:
                st.write("Matched ingredients:")
                st.write(", ".join(matched))

def render_backend_response(response_type, message, data):
    if response_type == "answer":
        st.markdown(message)

    elif response_type == "error":
        st.error(message)

    elif response_type == "recipe_ingested":
        st.success(message)
        # if data:
        #     st.caption("Recipe details")
        #     st.json(data)

    elif response_type == "user_created":
        st.success(message)

    elif response_type == "pantry_updated":
        st.success(message)

    elif response_type == "pantry_list":
        st.info(message)
        for item in data.get("pantry", []):
            st.write(f"- {item['name']} ({item['quantity']} {item['unit']})")

    elif response_type == "missing_ingredients":
        st.subheader(message)
        for ing in data.get("ingredients", []):
            st.write(f"- {ing}")

    elif response_type == "recipes_list":
        render_recipes(data.get("recipes", []))

    elif response_type == "course_plan":
        st.subheader(message)

        def extract_recipes(block):
            # case 1: ToolResult-like dict
            if isinstance(block, dict) and "data" in block:
                return block["data"].get("recipes", [])
            # case 2: already a list of recipes
            if isinstance(block, list):
                return block
            return []

        st.markdown("### Appetizer")
        render_recipes(extract_recipes(data.get("appetizer")))

        st.markdown("### Main")
        render_recipes(extract_recipes(data.get("main")))

        st.markdown("### Dessert")
        render_recipes(extract_recipes(data.get("dessert")))

    elif response_type == "rag_chunks":
        st.subheader(message)
        for c in data.get("chunks", []):
            st.write(c.get("text", ""))

    else:
        st.markdown(message or "Action completed.")

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

# def ensure_user():
#     if "user_id" not in st.session_state or st.session_state.user_id is None:
#         try:
#             r = requests.post(
#                 f"{BACKEND_URL}/agent/message",
#                 data={"message": "Create user"},
#                 timeout=30
#             )
#             r.raise_for_status()
#             resp = r.json()

#             if resp.get("type") == "user_created":
#                 st.session_state.user_id = resp["data"]["user_id"]
#             else:
#                 st.error("Failed to create user")

#         except Exception as e:
#             st.error(f"User creation failed: {e}")

def ensure_user():
    st.session_state.user_id = DEFAULT_USER_ID


def ensure_chat_exists():
    if st.session_state.current_chat_id is None:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "name": "Recipe upload",
            "messages": [],
            "created_at": datetime.now().isoformat()
        }
        st.session_state.current_chat_id = new_id

def add_system_message(text: str):
    chat = st.session_state.chats[st.session_state.current_chat_id]
    chat["messages"].append({
        "role": "assistant",
        "content": text
    })

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.chats = load_chats()
    st.session_state.current_chat_id = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "user_id" not in st.session_state:
    st.session_state.user_id = None

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Chat", "Add Ingredients", "Feedback"])

if page == "Chat":
    st.sidebar.markdown("### Chats")
    if st.sidebar.button("Start new chat"):
        st.session_state.current_chat_id = None
        st.session_state.processed_files.clear()
        st.session_state.pdf_uploader_key += 1
        st.session_state.image_uploader_key += 1

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
    # st.markdown("### ➕ Add recipe")

    col_pdf, col_img = st.columns(2)

    with col_pdf:
        pdf_file = st.file_uploader(
            "➕ Upload recipe from PDF",
            type=["pdf"],
            key=f"pdf_uploader_{st.session_state.pdf_uploader_key}"
        )
 
    with col_img:
        image_file = st.file_uploader(
            "➕ Upload recipe from image",
            type=["png", "jpg", "jpeg"],
            key=f"image_uploader_{st.session_state.image_uploader_key}"
        )

    if pdf_file is not None and pdf_file.name not in st.session_state.processed_files:
        with st.spinner("Processing recipe… this may take up to 1 minute"):
            try:
                ensure_user()

                tmp_dir = "/tmp/foodify_uploads"
                os.makedirs(tmp_dir, exist_ok=True)

                pdf_path = os.path.join(
                    tmp_dir,
                    f"{uuid.uuid4()}_{pdf_file.name}"
                )

                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getvalue())

                r = requests.post(
                    f"{BACKEND_URL}/agent/message",
                    data={
                        "user_id": st.session_state.user_id,
                        "message": f"Add recipe from PDF: {pdf_path}"
                    },
                    timeout=90
                )
                r.raise_for_status()
                resp = r.json()

                ensure_chat_exists()

                # DODAJ WIADOMOŚĆ DO CZATU
                success_msg = f"Recipe added from PDF: **{pdf_file.name}**"
                add_system_message(success_msg)

                save_chats(st.session_state.chats)

                # NIE RÓB st.rerun()

                st.success(success_msg)
                st.session_state.processed_files.add(pdf_file.name)
                st.session_state.pdf_uploader_key += 1

            except Exception as e:
                st.error(f"PDF upload failed: {e}")

    if image_file is not None and image_file.name not in st.session_state.processed_files:
        with st.spinner("Processing image recipe… this may take up to 1 minute"):
            try:
                ensure_user()

                tmp_dir = "/tmp/foodify_uploads"
                os.makedirs(tmp_dir, exist_ok=True)

                image_path = os.path.join(
                    tmp_dir,
                    f"{uuid.uuid4()}_{image_file.name}"
                )

                with open(image_path, "wb") as f:
                    f.write(image_file.getvalue())

                r = requests.post(
                    f"{BACKEND_URL}/agent/message",
                    data={
                        "user_id": st.session_state.user_id,
                        "message": f"Add recipe from image: {image_path}"
                    },
                    timeout=120
                )
                r.raise_for_status()
                resp = r.json()

                ensure_chat_exists()

                success_msg = f"Recipe added from image: **{image_file.name}**"
                add_system_message(success_msg)

                save_chats(st.session_state.chats)

                st.success(success_msg)
                st.session_state.processed_files.add(image_file.name)
                st.session_state.image_uploader_key += 1   

            except Exception as e:
                st.error(f"Image upload failed: {e}")

    if st.session_state.current_chat_id:
        chat = st.session_state.chats[st.session_state.current_chat_id]
        for msg in chat["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Type your message")
    
    if prompt:
        ensure_chat_exists()

        chat = st.session_state.chats[st.session_state.current_chat_id]

        safe_prompt = sanitize_message(prompt)

        response_type = None
        message = ""
        data = {}
        chat["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            ensure_user()

            r = requests.post(
                f"{BACKEND_URL}/agent/message",
                data={
                    "message": safe_prompt,
                    "user_id": st.session_state.user_id
                },
                timeout=30
            )

            r.raise_for_status()
            backend_response = r.json()

            response_type = backend_response.get("type")
            message = backend_response.get("message")
            data = backend_response.get("data") or {}

            with st.chat_message("assistant"):
                if response_type:
                    render_backend_response(response_type, message, data)
                else:
                    st.error("No response from backend.")

            if message and message.strip():
                chat["messages"].append({"role": "assistant", "content": message})

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Backend error: {e}")
            chat["messages"].append({"role": "assistant", "content": f"Backend error: {e}"})

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