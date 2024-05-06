from http import HTTPStatus
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from app import app

TEST_FILES = Path(__file__).parent / "testFiles"
TEST_FILES_INPUTS = TEST_FILES / "input_data"


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client: FlaskClient) -> None:
    response = client.get("/")
    assert response.status_code == 200


def test_upload_inventory_file(client: FlaskClient) -> None:
    with (TEST_FILES_INPUTS / "sklad.csv").open(mode="rb") as f:
        data = {"inventory_file": (f, "sklad.csv")}
        response = client.post("/upload_inventory", data=data, content_type="multipart/form-data")
    assert response.status_code == HTTPStatus.OK
    assert "success" in response.json


def test_upload_demand_file(client: FlaskClient) -> None:
    with (TEST_FILES_INPUTS / "spotřeba.csv").open(mode="rb") as f:
        data = {"demand_file": (f, "spotřeba.csv")}
        response = client.post("/upload_demand", data=data, content_type="multipart/form-data")
    assert response.status_code == HTTPStatus.OK
    assert "success" in response.json


def test_upload_discount_file(client: FlaskClient) -> None:
    with (TEST_FILES_INPUTS / "rabaty.csv").open(mode="rb") as f:
        data = {"price_discount_file": (f, "rabaty.csv")}
        response = client.post("/upload_price_discount", data=data, content_type="multipart/form-data")
    assert response.status_code == HTTPStatus.OK
    assert "success" in response.json


def test_discount_file_format(client: FlaskClient) -> None:
    with (TEST_FILES_INPUTS / "rabaty_spatne.csv").open(mode="rb") as f:
        data = {"price_discount_file": (f, "rabaty_spatne.csv")}
        response = client.post("/upload_price_discount", data=data, content_type="multipart/form-data")
    assert response.status_code == HTTPStatus.OK
    assert response.json["message"] == "Uploaded file must contain only numerical values"


def test_upload_demand_file_no_file(client: FlaskClient) -> None:
    data = {}
    response = client.post("/upload_demand", data=data, content_type="multipart/form-data")
    assert response.status_code == HTTPStatus.OK
    assert "success" in response.json
    assert response.json["success"] == False
    assert "message" in response.json
    assert response.json["message"] == "No file uploaded"


def test_upload_inventory_file_no_file(client: FlaskClient) -> None:
    data = {}
    response = client.post("/upload_inventory", data=data, content_type="multipart/form-data")
    assert response.status_code == HTTPStatus.OK
    assert "success" in response.json
    assert response.json["success"] == False
    assert "message" in response.json
    assert response.json["message"] == "No file uploaded"
