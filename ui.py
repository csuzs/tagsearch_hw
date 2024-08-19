import json
import os
from typing import List

import gradio as gr
import pandas as pd
import requests

from create_dummy_tags import tags

# Define the URL for the FastAPI service
BASE_URL = os.environ.get("API_BASE_URL", default="http://app:8000")


def upload_csv(file):
    filename = file.name
    with open(filename, "rb") as f:
        files = {"file": (filename, f, "text/csv")}
        response = requests.post(f"{BASE_URL}/upload-csv/", files=files)
        if response.status_code == 200:
            return response.json()["message"]
        else:
            return response.json()["detail"]


def search_names(query, k=5):
    if query == "":
        raise gr.Error("Please enter a search query")

    if k < 1:
        raise gr.Error("Number of results should be at least 1")

    response = requests.get(f"{BASE_URL}/search/", params={"query": query, "k": k})
    if response.status_code == 200:
        data = response.json()
        semantic_results = data["match"]["semantic"]
        typo_results = data["match"]["typo"]

        semantic_df = pd.DataFrame(semantic_results)
        typo_df = pd.DataFrame(typo_results)

        return semantic_df, typo_df
    else:
        return response.json()["detail"], pd.DataFrame()


def delete_collection():
    response = requests.delete(f"{BASE_URL}/purge/")
    if response.status_code == 200:
        return response.json()["message"]
    else:
        return response.json()["detail"]


def check_task_status():
    response = requests.get(f"{BASE_URL}/task-status/")
    if response.status_code == 200:
        resp_str = json.dumps(response.json(), indent=4)
        return resp_str
    else:
        return "Something bad happened"


def add_synonym(tag_1: str, tag_2: str):
    print("tag1:", tag_1)
    print("tag2:", tag_2)
    if not tag_1 or not tag_2:
        raise gr.Error("Please select synonym tags")

    response = requests.post(
        f"{BASE_URL}/add_synonym/", params={"tag_1": tag_1, "tag_2": tag_2}
    )
    if response.status_code == 200:
        return str(response.content)
    else:
        return "Something bad happened"


def update_tag_list_dropdown():
    response = requests.get(f"{BASE_URL}/get_tag_list/")
    if response.status_code == 200:
        return response.json()["tag_list"]
    else:
        gr.Error("someting bad happened")


# Gradio interface
with gr.Blocks(
    css="body { font-family: Arial, sans-serif; } footer { visibility: hidden }"
) as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>Tag Search API</h1><h3 style='text-align: center;'>by the AI Team</h3>"
    )

    with gr.Tab("Search"):
        with gr.Row():
            query = gr.Textbox(
                label="Search Query",
                placeholder="Enter tag to search",
                elem_classes="tab-container",
            )
            k = gr.Number(
                label="Number of Results", value=5, elem_classes="tab-container"
            )
        search_button = gr.Button("Search", elem_classes="tab-container")
        semantic_table = gr.Dataframe(
            label="Semantic Matches", interactive=False, elem_classes="tab-container"
        )
        typo_table = gr.Dataframe(
            label="Typo Matches", interactive=False, elem_classes="tab-container"
        )

        search_button.click(
            fn=search_names, inputs=[query, k], outputs=[semantic_table, typo_table]
        )
        query.submit(
            fn=search_names, inputs=[query, k], outputs=[semantic_table, typo_table]
        )

    with gr.Tab("Upload CSV"):
        with gr.Accordion("Information", open=False):
            gr.Markdown(
                """## Info
                        - the csv file must be **comma separated**
                        - the csv file **should have a 'name' column**, anything else will be discarded
                        - processing time depends on the number of records, you can check the status with the button below
                        """
            )
        csv_file = gr.File(
            label="Upload CSV File", type="filepath", elem_classes="tab-container"
        )
        upload_button = gr.Button("Upload", elem_classes="tab-container")
        upload_status = gr.Textbox(
            label="Upload Status", interactive=False, elem_classes="tab-container"
        )

        upload_button.click(fn=upload_csv, inputs=csv_file, outputs=upload_status)

        status_button_upload = gr.Button(
            "Check Task Status", elem_classes="tab-container"
        )
        status_display_upload = gr.Textbox(
            label="Task Status", interactive=False, elem_classes="tab-container"
        )
        status_button_upload.click(fn=check_task_status, outputs=status_display_upload)

        delete_button = gr.Button("Delete Collection", elem_classes="tab-container")
        delete_status = gr.Textbox(
            label="Delete Status", interactive=False, elem_classes="tab-container"
        )
        delete_button.click(fn=delete_collection, outputs=delete_status)

    with gr.Tab("Add Synonym"):

        #!TODO load tag list dynamically from backend - not proud of this solution, but no idea why app.name_storage doesn't get filled with values on backend
        list_dropdown_1 = gr.Dropdown(elem_classes="tab-container", choices=tags)
        list_dropdown_2 = gr.Dropdown(elem_classes="tab-container", choices=tags)
        # list_dropdown_1.select(fn=update_tag_list_dropdown,outputs=[list_dropdown_1])
        # list_dropdown_2.select(fn=update_tag_list_dropdown,outputs=[list_dropdown_2])

        update_button = gr.Button("Update tag list", elem_classes="tab-container")
        update_status = gr.Textbox(
            label="synonym update status",
            interactive=False,
            elem_classes="tab-container",
        )
        update_button.click(
            fn=add_synonym,
            inputs=[list_dropdown_1, list_dropdown_2],
            outputs=update_status,
        )


# Launch the interface
demo.launch(server_name="0.0.0.0")
