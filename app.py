import base64
import io
import math
import os
import urllib
from dataclasses import dataclass, field
from typing import Hashable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import statsmodels.api as sm
from flask import *
from scipy import stats

matplotlib.use("Agg")
from werkzeug.utils import secure_filename

UPLOAD_FOLDER_DEMAND = os.path.join("uploadFiles", "uploads", "demand")
UPLOAD_FOLDER_INVENTORY = os.path.join("uploadFiles", "uploads", "inventory")
UPLOAD_FOLDER_QUANTITY_DISCOUNT = os.path.join("uploadFiles", "uploads", "discount")
RESULT_INVENTORY = os.path.join("uploadFiles", "result")
# ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config["UPLOAD_FOLDER_DEMAND"] = UPLOAD_FOLDER_DEMAND
app.config["UPLOAD_FOLDER_INVENTORY"] = UPLOAD_FOLDER_INVENTORY
app.config["UPLOAD_FOLDER_QUANTITY_DISCOUNT"] = UPLOAD_FOLDER_QUANTITY_DISCOUNT
app.config["RESULT_INVENTORY"] = RESULT_INVENTORY
app.secret_key = "This is your secret key to utilize session in Flask"


@dataclass(frozen=True)
class Product:
    name: Hashable
    price: float
    storage_cost: float


@dataclass
class Order:
    delivery_fee: float
    discount_rate: float
    extra_costs: float = 0

    products: dict[Product, int] = field(default_factory=dict)

    def add_product(self, name: Hashable, price: float, storage_cost: float, quantity: int) -> None:
        product = Product(name, price, storage_cost)
        self.products[product] = quantity

    @property
    def quantities(self) -> dict[str, int]:
        return {str(product.name): quantity for product, quantity in self.products.items()}

    @property
    def order_costs_with_discount_applied(self) -> float:
        order_costs = self.order_costs
        if self.discount_rate:
            return order_costs * (1 - (self.discount_rate / 100))
        return order_costs

    @property
    def order_costs(self) -> float:
        return sum([quantity * product.price for product, quantity in self.products.items()])

    @property
    def storage_costs(self) -> float:
        return sum([quantity * product.storage_cost for product, quantity in self.products.items()])

    @property
    def total_costs(self) -> float:
        return self.order_costs_with_discount_applied + self.storage_costs + self.extra_costs + self.delivery_fee


# / cesta aplikace
@app.route("/")
def home():
    for filename in os.listdir(app.config["RESULT_INVENTORY"]):
        file_to_delete = os.path.join(app.config["RESULT_INVENTORY"], filename)
        try:
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)
                print(f"Successfully deleted file at {file_to_delete}")
        except Exception as e:
            print(f"Error deleting file at {file_to_delete}: {e}")

    for filename in os.listdir(app.config["UPLOAD_FOLDER_INVENTORY"]):
        file_to_delete = os.path.join(app.config["UPLOAD_FOLDER_INVENTORY"], filename)
        try:
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)
                print(f"Successfully deleted file at {file_to_delete}")
        except Exception as e:
            print(f"Error deleting file at {file_to_delete}: {e}")

    for filename in os.listdir(app.config["UPLOAD_FOLDER_QUANTITY_DISCOUNT"]):
        file_to_delete = os.path.join(app.config["UPLOAD_FOLDER_QUANTITY_DISCOUNT"], filename)
        try:
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)
                print(f"Successfully deleted file at {file_to_delete}")
        except Exception as e:
            print(f"Error deleting file at {file_to_delete}: {e}")

    for filename in os.listdir(app.config["UPLOAD_FOLDER_DEMAND"]):
        file_to_delete = os.path.join(app.config["UPLOAD_FOLDER_DEMAND"], filename)
        try:
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)
                print(f"Successfully deleted file at {file_to_delete}")
        except Exception as e:
            print(f"Error deleting file at {file_to_delete}: {e}")
    session.pop("percentage_data", None)
    session.pop("delivery_data", None)
    session.pop("service_data", None)
    session.pop("delivery_fee_data", None)
    return render_template("index.html")


@app.route("/upload_inventory", methods=["POST"])
def uploadInventoryFile():
    if "inventory_file" in request.files:
        inventory_file = request.files["inventory_file"]
        inventory_filename = secure_filename(inventory_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER_INVENTORY"], inventory_filename)

        # Delete all existing files in the inventory folder
        for filename in os.listdir(app.config["UPLOAD_FOLDER_INVENTORY"]):
            file_to_delete = os.path.join(app.config["UPLOAD_FOLDER_INVENTORY"], filename)
            try:
                if os.path.isfile(file_to_delete):
                    os.remove(file_to_delete)
                    print(f"Successfully deleted file at {file_to_delete}")
            except Exception as e:
                print(f"Error deleting file at {file_to_delete}: {e}")

        # Save the new file
        inventory_file.save(file_path)
        session["uploaded_data_file_path_inventory"] = file_path

        try:
            df = pd.read_csv(file_path, delimiter=";", header=None)
            if len(df.columns) != 3:
                os.remove(file_path)  # Delete the file if not valid
                return jsonify({"success": False, "message": "Uploaded file must have exactly three columns"})
        except Exception as e:
            os.remove(file_path)  # Delete the file if not valid
            return jsonify({"success": False, "message": f"Error reading file: {e}"})

        return jsonify({"success": True, "message": "File uploaded successfully"})
    else:
        return jsonify({"success": False, "message": "No file uploaded"})


# nahrání souboru s diskontními rabaty
@app.route("/upload_price_discount", methods=["GET", "POST"])
def uploadPriceDiscountFile():
    if request.method == "POST":
        if "price_discount_file" in request.files:
            price_discount_file = request.files["price_discount_file"]
            price_discount_filename = secure_filename(price_discount_file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER_QUANTITY_DISCOUNT"], price_discount_filename)
            for filename in os.listdir(app.config["UPLOAD_FOLDER_QUANTITY_DISCOUNT"]):
                file_to_delete = os.path.join(app.config["UPLOAD_FOLDER_QUANTITY_DISCOUNT"], filename)
                try:
                    if os.path.isfile(file_to_delete):
                        os.remove(file_to_delete)
                        print(f"Successfully deleted file at {file_to_delete}")
                except Exception as e:
                    print(f"Error deleting file at {file_to_delete}: {e}")

            # Save the new file
            price_discount_file.save(file_path)
            session["uploaded_data_file_path_price_discount"] = file_path
        try:
            df = pd.read_csv(file_path, delimiter=";", header=None)
            if not df.apply(lambda x: x.map(lambda y: isinstance(y, (int, float)))).all().all():
                os.remove(file_path)
                return jsonify({"success": False, "message": "Uploaded file must contain only numerical values"})
            if len(df.columns) != 3:
                os.remove(file_path)
                return jsonify({"success": False, "message": "Uploaded file must have exactly three columns"})
        except Exception as e:
            os.remove(file_path)
            return jsonify({"success": False, "message": f"Error reading file: {e}"})

        return jsonify({"success": True, "message": "File uploaded successfully"})
    else:
        return jsonify({"success": False, "message": "No file uploaded"})


# nahrání souboru s poptávkou
@app.route("/upload_demand", methods=["GET", "POST"])
def uploadDemandFile():
    if request.method == "POST":
        if "demand_file" in request.files:
            demand_file = request.files["demand_file"]
            demand_filename = secure_filename(demand_file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER_DEMAND"], demand_filename)

            # Delete all existing files in the demand folder
            for filename in os.listdir(app.config["UPLOAD_FOLDER_DEMAND"]):
                file_to_delete = os.path.join(app.config["UPLOAD_FOLDER_DEMAND"], filename)
                try:
                    if os.path.isfile(file_to_delete):
                        os.remove(file_to_delete)
                        print(f"Successfully deleted file at {file_to_delete}")
                except Exception as e:
                    print(f"Error deleting file at {file_to_delete}: {e}")

            # Save the new file
        demand_file.save(file_path)
        session["uploaded_data_file_path_demand"] = file_path
        try:
            df = pd.read_csv(file_path, delimiter=";", header=None)
            if len(df.columns) < 2:
                os.remove(file_path)  
                return jsonify({"success": False, "message": "File must have at least 2 columns"})

            for i in range(1, len(df.columns)):
                col_values = pd.to_numeric(df.iloc[:, i], errors="coerce")
                if col_values.isna().any():
                    os.remove(file_path)
                    return jsonify({"success": False, "message": f"Values in column {i+1} must be numeric"})

        except Exception as e:
            os.remove(file_path)  
            return jsonify({"success": False, "message": f"Error reading file: {e}"})

        return jsonify({"success": True, "message": "File uploaded successfully"})
    else:
        return jsonify({"success": False, "message": "No file uploaded"})


# nahrání pořizovacích nákladů
@app.route("/submit_delivery_fee", methods=["GET", "POST"])
def submit_delivery_fee():
    if request.method == "POST":
        fee_value = request.json.get("feeValue")
        session["delivery_fee_data"] = fee_value
        print("Delivery fee value received:", fee_value)
        return jsonify({"success": True, "message": "Delivery fee uploaded successfully!"})
    else:
        return jsonify({"success": False, "message": "Method not allowed."})


# nahrání úrokové míry
@app.route("/submit_percentage", methods=["GET", "POST"])
def submit_percentage_rate():
    if request.method == "POST":
        float_value = request.json.get("floatValue")
        session["percentage_data"] = float_value

        print("Float value received:", float_value)

        return jsonify({"success": True, "message": "Percentage Rate of Charge uploaded successfully!"})
    else:
        return jsonify({"success": False, "message": "Method not allowed."})


@app.route("/submit_delivery", methods=["GET", "POST"])
def submit_delivery():
    if request.method == "POST":
        # Retrieve the float value from the JSON data of the POST request
        integer_value = request.json.get("integerValue")
        session["delivery_data"] = integer_value

        print("Float value received:", integer_value)

        return jsonify({"success": True, "message": "Percentage Rate of Charge uploaded successfully!"})
    else:
        return jsonify({"success": False, "message": "Method not allowed."})


# nahrání požadované míry obsluhy
@app.route("/submit_service", methods=["GET", "POST"])
def submit_service():
    if request.method == "POST":
        service_value = request.json.get("serviceValue")
        session["service_data"] = service_value

        print("Float value received:", service_value)

        return jsonify({"success": True, "message": "Percentage Rate of Charge uploaded successfully!"})
    else:
        return jsonify({"success": False, "message": "Method not allowed."})


# zobrazení instrukcí v češtině
@app.route("/show_instructions_czech")
def showInstructions():
    return render_template("navod_cs.html")


# zobrazení instrukcí v angličtině
@app.route("/show_instructions_english")
def showInstructionsEN():
    return render_template("navod_en.html")


# zobrazení aplikace
@app.route("/show_app")
def showApp():
    return render_template("index.html")


# zobrazení nahraných dat o skladových zásobách
@app.route("/show_data_inventory")
def showDataInventory():
    # Uploaded File Path
    data_file_path = session.get("uploaded_data_file_path_inventory", None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding="UTF-8", delimiter=";", header=None)
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template("show_csv_data.html", data_var=uploaded_df_html)


# zobrazení nahraných dat o poptávce
@app.route("/show_data_demand")
def showDataDemand():
    # Uploaded File Path
    data_file_path = session.get("uploaded_data_file_path_demand", None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding="UTF-8", delimiter=";", header=None)
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template("show_csv_data.html", data_var=uploaded_df_html)


# zobrazení nahraných dat o diskontních rabatech
@app.route("/show_data_discount")
def showDataDiscount():
    # Uploaded File Path
    data_file_path = session.get("uploaded_data_file_path_price_discount", None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding="UTF-8", delimiter=";", header=None)
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template("show_csv_data.html", data_var=uploaded_df_html)


# zobrazení nahraných dat o určení rozdělení
@app.route("/calculate_distribution")
def calculate_distribution():
    data_file_path = session.get("uploaded_data_file_path_demand", None)
    if data_file_path is not None:
        data = pd.read_csv(data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)
        np.random.seed(42)
        distributions = [stats.norm, stats.poisson, stats.uniform]
        distribution_result = {}

        for product, values in data.iterrows():
            best_fit = None
            best_p_value = 0
            for distribution in distributions:
                if distribution.name == "norm":
                    sample = distribution.rvs(loc=np.mean(values), scale=np.std(values, ddof=1), size=len(values))
                    _, p_value = stats.shapiro(values)
                elif distribution.name == "poisson":
                    sample = distribution.rvs(mu=np.mean(values), size=len(values))
                    _, p_value = stats.kstest(values, distribution.name, args=(sample,))
                else:
                    sample = distribution.rvs(
                        loc=np.min(values), scale=np.max(values) - np.min(values), size=len(values)
                    )
                    _, p_value = stats.kstest(values, distribution.name, args=(sample,))
                if p_value > best_p_value:
                    best_p_value = p_value
                    best_fit = distribution.name

            distribution_result[product] = best_fit

        return jsonify({"distribution_result": distribution_result})
    else:
        return jsonify({"error": "No data file uploaded"})


# výpočet skladovacích nákladů


def calculate_storage_costs():
    data_file_path = session.get("uploaded_data_file_path_inventory", None)
    if data_file_path is None:
        return None, None
    else:
        data = pd.read_csv(data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)

    if "percentage_data" not in session:
        return None, None

    percentage_rate = float(session.get("percentage_data", None))

    storage_costs = {}
    storage_costs_calc = {}
    prices = {}

    for product, row in data.iterrows():
        quantity = float(row[1])
        price = float(row[2])
        prices[product] = float(price)
        storage_costs[product] = quantity * price * (percentage_rate / 100)
        storage_costs_calc[product] = price * (percentage_rate / 100)

    return storage_costs, storage_costs_calc, prices


@app.route("/storage_costs", methods=["GET"])
def storage_costs():
    storage_costs, storage_costs_calc, prices = calculate_storage_costs()
    if storage_costs is None or prices is None or storage_costs_calc is None:
        return jsonify(message="No inventory file uploaded or percentage rate is not set in the session"), 400
    return jsonify(storage_costs=storage_costs, prices=prices, storage_costs_calc=storage_costs_calc)


# výpočet statistických hodnot
def calculate_statistics():
    data_file_path = session.get("uploaded_data_file_path_demand", None)
    statistics = {}
    statistics_show = {}
    consumption_per_product = {}
    total_consumption = 0
    if data_file_path is not None:
        data = pd.read_csv(data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)
        for product, values in data.iterrows():
            delivery = float(session.get("delivery_data"))
            mean = np.mean(values) * delivery
            mean_calc = np.mean(values)
            std_dev = np.std(values, ddof=1) * delivery
            mean_show = math.ceil(np.mean(values) * delivery)
            std_dev_show = round((np.std(values, ddof=1) * delivery), 2)
            statistics[product] = {"mean": mean, "std_dev": std_dev, "mean_calc": mean_calc}
            statistics_show[product] = {"mean_show": mean_show, "std_dev_show": std_dev_show}
            consumption_for_product = sum(values)
            consumption_per_product[product] = {"consumption_for_product": consumption_for_product}
            total_consumption += sum(values)
    print(f"Konzumace: {total_consumption}")
    print(f"odchylka {product}: {statistics}")
    print(f"Consumption per product: {consumption_per_product}")
    return statistics, total_consumption, consumption_per_product, statistics_show


# volání výpočtu statistických hodnot
@app.route("/calculate_statistics", methods=["GET"])
def calculate_statistics_route():
    statistics, total_consumption, consumption_per_product, statistics_show = calculate_statistics()
    return jsonify(
        statistics=statistics,
        total_consumption=total_consumption,
        consumption_per_product=consumption_per_product,
        statistics_show=statistics_show,
    )


# výpočet úrovně znovuobjednávky při zadané úrovni obsluhy
def calculate_service_quantity():
    data_file_path = session.get("uploaded_data_file_path_demand", None)
    stock_data_file_path = session.get("uploaded_data_file_path_inventory", None)
    percentage_rate = float(session.get("percentage_data", None))
    statistics = {}
    reorder_point = {}
    total_storage_costs = 0
    original_storage_costs = 0
    safety_stock_costs = 0

    if data_file_path is not None and stock_data_file_path is not None:
        data = pd.read_csv(data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)
        stock_data = pd.read_csv(stock_data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)

        for product, values in data.iterrows():
            delivery = float(session.get("delivery_data"))
            mean = math.ceil(np.mean(values) * delivery)
            std_dev = np.std(values, ddof=1) * delivery
            service_level = float(session.get("service_data")) / 100
            z = stats.norm(0, 1).ppf(service_level)
            safety_stock = z * std_dev
            safety_stock_calc = math.ceil(safety_stock)
            print(f"safety stock for product {product} {safety_stock_calc}")
            level_quantity = math.ceil(mean + safety_stock)

            statistics[product] = {"level_quantity": level_quantity}

            reorder_quantity = level_quantity + mean
            reorder_point[product] = {"reorder_quantity": reorder_quantity}

            if product in stock_data.index:
                price = stock_data.loc[product, 2]
                original_quantity = stock_data.loc[product, 1]
                total_storage_costs += level_quantity * price * percentage_rate / 100
                safety_stock_costs += safety_stock_calc * price * percentage_rate / 100
                original_storage_costs += original_quantity * price * percentage_rate / 100
                cost_difference = total_storage_costs - original_storage_costs
                print(f"safety stock costs{safety_stock_costs}")

    return statistics, reorder_point, safety_stock_costs, total_storage_costs, cost_difference


@app.route("/service_quantity", methods=["GET"])
def service_quantity():
    statistics, reorder_point, safety_stock_costs, total_storage_costs, cost_difference = calculate_service_quantity()
    return jsonify(
        statistics=statistics,
        reorder_point=reorder_point,
        safety_stock_costs=safety_stock_costs,
        total_storage_costs=total_storage_costs,
        cost_difference=cost_difference,
    )


@app.route("/calculate_optimal_discount_quantity", methods=["GET"])
def optimal_discount_quantity():
    language = request.args.get("language", "English")

    stock_data_file_path = session.get("uploaded_data_file_path_inventory", None)
    discount_data_file_path = session.get("uploaded_data_file_path_price_discount", None)
    demand_data_file_path = session.get("uploaded_data_file_path_demand", None)

    stock_data = pd.read_csv(stock_data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)
    discount_data = pd.read_csv(discount_data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)
    demand_data = pd.read_csv(demand_data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)

    if language == "Czech":
        quantities_columns = ["Produkt", "Optimální množství"]
        costs_columns = ["Položka", "Hodnota"]
        items = [
            "Dopravné",
            "Součet skladovacích nákladů",
            "Cena samotné objednávky (se slevou)",
            "Použitá sleva",
            "Suma celkových nákladů",
        ]
        quantities_sheet_name = "Optimální Množství"
        costs_sheet_name = "Přehled Nákladů"

    else:
        quantities_columns = ["Product", "Optimal Quantity"]
        costs_columns = ["Item", "Value"]
        items = ["Delivery Fee", "Sum Storage Costs", "Order Cost", "Discount Rate used", "Sum of Total Costs"]
        quantities_sheet_name = "Optimal Quantities"
        costs_sheet_name = "Costs Overview"

    percentage_rate = float(session.get("percentage_data", None))
    delivery_fee = float(session.get("delivery_fee_data", None))
    total_consumption = demand_data.sum().sum()  # Total consumption across all products

    consumption_per_product = {}
    statistics = {}
    optimal_quantities = {}

    optimal_order = None

    for product, values in demand_data.iterrows():
        mean = np.mean(values)
        std_dev = np.std(values, ddof=1)
        statistics[product] = {"mean": mean, "std_dev": std_dev}

        consumption_for_product = sum(values)
        consumption_per_product[product] = {"consumption_for_product": consumption_for_product}

    for lower_bound, upper_bound, discount_rate in discount_data.itertuples():
        order = Order(delivery_fee=delivery_fee, discount_rate=discount_rate)

        for product, (_, price) in stock_data.iterrows():
            mean = float(statistics[product]["mean"])
            consumption_for_each = float(consumption_per_product[product]["consumption_for_product"])

            storage_cost = price * (percentage_rate / 100)
            if discount_rate:
                storage_cost *= 1 - (discount_rate / 100)

            quantity = math.sqrt(
                (2 * mean * ((delivery_fee * consumption_for_each) / total_consumption)) / storage_cost,
            )
            quantity_rounded = math.ceil(quantity)

            order.add_product(product, price, storage_cost, quantity_rounded)

        if order.order_costs > upper_bound:
            continue

        if order.order_costs < lower_bound:
            order.extra_costs += lower_bound - order.order_costs

        if (not optimal_order) or (order.total_costs < optimal_order.total_costs):
            optimal_order = order

    if not optimal_order:
        return None

    # Create a DataFrame for optimal quantities
    optimal_quantities = [[product.name, quantity] for product, quantity in optimal_order.products.items()]
    df_optimal_quantities = pd.DataFrame(optimal_quantities, columns=quantities_columns)

    total_costs = {
        costs_columns[0]: items,
        costs_columns[1]: [
            optimal_order.delivery_fee,
            optimal_order.storage_costs,
            optimal_order.order_costs_with_discount_applied,
            optimal_order.discount_rate,
            optimal_order.total_costs,
        ],
    }
    df_total_costs = pd.DataFrame(total_costs, columns=costs_columns)

    result_filename = "result.xlsx"
    excel_file_path = os.path.join(app.config["RESULT_INVENTORY"], result_filename)

    with pd.ExcelWriter(excel_file_path) as writer:
        df_optimal_quantities.to_excel(writer, sheet_name=quantities_sheet_name, index=False)
        df_total_costs.to_excel(writer, sheet_name=costs_sheet_name, index=False)

    print(f"Data saved to {excel_file_path}")
    return jsonify(
        optimal_quantities=optimal_order.quantities,
        best_total_cost=optimal_order.total_costs,
        excel_file_path=excel_file_path,
        used_rate=optimal_order.discount_rate,
    )


# výpočet optimálního množství bez rabatu
@app.route("/calculate_optimal_quantity", methods=["GET"])
def optimal_quantity():
    language = request.args.get("language", "English")

    _, storage_costs_calc, prices = calculate_storage_costs()
    if prices is None or storage_costs_calc is None:
        return jsonify(message="Error retrieving storage costs"), 400

    statistics, total_consumption, consumption_per_product, _ = calculate_statistics()
    if statistics is None:
        return jsonify(message="Error retrieving statistics"), 400

    quantities = {}
    total_cost = 0
    delivery_fee = float(session.get("delivery_fee_data", None))
    sum_storage_costs = 0

    if language == "Czech":
        quantities_columns = ["Produkt", "Množství"]
        costs_columns = ["Položka", "Hodnota"]
        items = ["Dopravné", "Součet skladovacích nákladů", "Cena samotné objednávky", "Součet celkových nákladů"]
        costs_sheet_name = "Přehled Nákladů"
        quantities_sheet_name = "Optimální Množství"
    else:
        quantities_columns = ["Product", "Quantity"]
        costs_columns = ["Item", "Value"]
        items = ["Delivery Fee", "Sum Storage Costs", "Order Cost", "Sum of total costs"]
        costs_sheet_name = "Costs Overview"
        quantities_sheet_name = "Optimal Quantities"

    for product in statistics.keys():
        storage_costs = float(storage_costs_calc[product])

        mean_calc = float(statistics[product]["mean_calc"])
        consumption_for_each = float(consumption_per_product[product]["consumption_for_product"])
        quantity = math.ceil(
            math.sqrt((2 * mean_calc * ((delivery_fee * consumption_for_each) / total_consumption)) / (storage_costs))
        )
        quantities[product] = quantity
        sum_storage_costs += storage_costs * quantities[product]
        total_cost += quantity * prices[product]

    sum_total_costs = delivery_fee + sum_storage_costs + total_cost
    quantities_df = pd.DataFrame(list(quantities.items()), columns=quantities_columns)

    costs_df = pd.DataFrame(
        {costs_columns[0]: items, costs_columns[1]: [delivery_fee, sum_storage_costs, total_cost, sum_total_costs]}
    )

    result_filename = "result.xlsx"
    excel_file_path = os.path.join(app.config["RESULT_INVENTORY"], result_filename)
    with pd.ExcelWriter(excel_file_path) as writer:
        quantities_df.to_excel(writer, sheet_name=quantities_sheet_name, index=False)
        costs_df.to_excel(writer, sheet_name=costs_sheet_name, index=False)

    print("Data saved to Excel:", excel_file_path)
    print("Total cost:", total_cost)

    return jsonify(quantities=quantities, total_cost=total_cost, excel_file_path=excel_file_path)


# vytvoření histogramu
@app.route("/histogram")
def create_histogram():
    language = request.args.get("language", "English")

    demand_data_file_path = session.get("uploaded_data_file_path_demand", None)
    demand_data = pd.read_csv(demand_data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)

    images = []
    for product, rows in demand_data.iterrows():
        plt.figure()
        plt.hist(rows, bins=10, edgecolor="black")

        if language == "Czech":
            plt.xlabel("Spotřeba")
            plt.ylabel("Počet dní")
            plt.title(f"Histogram spotřeby {product}")
        else:
            plt.xlabel("Consumption")
            plt.ylabel("Number of days")
            plt.title(f"Consumption histogram for {product}")

        # Convert plot to PNG image
        png_image = io.BytesIO()
        plt.savefig(png_image, format="png")
        png_image.seek(0)
        png_image_b64_string = "data:image/png;base64,"
        png_image_b64_string += urllib.parse.quote(base64.b64encode(png_image.read()))

        images.append(png_image_b64_string)

    # Return the base64 encoded image
    return jsonify({"image_data": images})


# vytvoření QQ grafu
@app.route("/qqplot")
def create_qqplot():
    language = request.args.get("language", "English")

    demand_data_file_path = session.get("uploaded_data_file_path_demand", None)
    demand_data = pd.read_csv(demand_data_file_path, encoding="UTF-8", delimiter=";", index_col=0, header=None)

    images = []
    for product, rows in demand_data.iterrows():
        plt.figure()
        sm.qqplot(rows, line="45", fit=True)

        if language == "Czech":
            plt.xlabel("Teoretické kvantily")
            plt.ylabel("Vzorkové kvantily")
            plt.title(f"QQ plot pro spotřebu {product}")
        else:
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Sample Quantiles")
            plt.title(f"QQ plot for consumption of {product}")

        # Convert plot to PNG image
        png_image = io.BytesIO()
        plt.savefig(png_image, format="png")
        png_image.seek(0)
        png_image_b64_string = "data:image/png;base64,"
        png_image_b64_string += urllib.parse.quote(base64.b64encode(png_image.read()))

        images.append(png_image_b64_string)

    # Return the base64 encoded image
    return jsonify({"image_data": images})


# vstažení souboru
@app.route("/download_file")
def download_xlsx():
    result_filename = "result.xlsx"
    excel_file_path = os.path.join(app.config["RESULT_INVENTORY"], result_filename)
    return send_file(excel_file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
