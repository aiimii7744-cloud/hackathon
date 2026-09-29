"""Synthetic relational data pipeline powered by Google Gemini (LangChain)."""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_MODEL = "gemini-3.6-flash"
OUTPUT_FILE = "synthetic_output.json"


# ---------------------------------------------------------------------
# 1. Pydantic schemas
# ---------------------------------------------------------------------

class ColumnDefinition(BaseModel):
    name: str = Field(..., description="Name of the column")
    data_type: str = Field(..., description="SQL data type (e.g., VARCHAR, INT, FLOAT, TIMESTAMP)")
    is_primary_key: bool = Field(False, description="Whether this column is a primary key")
    is_nullable: bool = Field(True, description="Whether this column can contain NULL values")
    constraints: Optional[str] = Field(None, description="Business rules or constraints (e.g., > 0, regex)")


class ForeignKeyRelation(BaseModel):
    from_table: str = Field(..., description="Table containing the foreign key")
    from_column: str = Field(..., description="Foreign key column name")
    to_table: str = Field(..., description="Referenced (parent) table")
    to_column: str = Field(..., description="Referenced primary key column")


class TableSchema(BaseModel):
    table_name: str = Field(..., description="Name of the table")
    columns: List[ColumnDefinition] = Field(..., description="Columns in this table")


class RelationalSchemaOutput(BaseModel):
    tables: List[TableSchema] = Field(..., description="All identified tables")
    relationships: List[ForeignKeyRelation] = Field(..., description="Foreign key relationships between tables")


class GeneratedTableData(BaseModel):
    table_name: str = Field(..., description="Target table name")
    records: List[Dict[str, Any]] = Field(..., description="Generated records as key-value pairs")


class SyntheticDatasetOutput(BaseModel):
    dataset: List[GeneratedTableData] = Field(..., description="Generated records grouped by table")


# ---------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------

def build_chain(llm: ChatGoogleGenerativeAI, output_model: type[BaseModel], system_prompt: str, human_prompt: str):
    """Build a prompt -> LLM -> Pydantic parser chain."""
    parser = PydanticOutputParser(pydantic_object=output_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nReturn ONLY valid JSON.\n{format_instructions}"),
        ("human", human_prompt),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def content_to_text(content: Any) -> str:
    """Convert an LLM response content (string or list of parts) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------
# 3. Schema extractor
# ---------------------------------------------------------------------

class SchemaExtractor:
    """Extracts a relational schema (tables, keys, relationships) from user input."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.chain = build_chain(
            llm,
            RelationalSchemaOutput,
            system_prompt=(
                "You are an expert database architect. Analyze the provided input "
                "(a database schema, sample CSV/JSON, or plain-text business rules) "
                "and produce a normalized relational schema with explicit primary and foreign keys."
            ),
            human_prompt="{user_input}",
        )

    def extract(self, user_input: str) -> RelationalSchemaOutput:
        return self.chain.invoke({"user_input": user_input})


# ---------------------------------------------------------------------
# 4. Synthetic data generator
# ---------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generates synthetic records that respect the schema and business rules."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.chain = build_chain(
            llm,
            SyntheticDatasetOutput,
            system_prompt=(
                "You are a synthetic data generation system. Using the database schema, "
                "relationships, and user instructions, generate realistic synthetic records.\n"
                "- Ensure referential integrity (foreign keys must match primary keys in parent tables).\n"
                "- Use realistic names, text, dates, and amounts, including realistic edge cases.\n"
                "- Generate the requested number of rows per table."
            ),
            human_prompt=(
                "Schema and relationships:\n{schema}\n\n"
                "Business rules and input:\n{user_input}\n\n"
                "Records to generate per table: {num_records}"
            ),
        )

    def generate(self, schema: RelationalSchemaOutput, user_input: str, num_records: int) -> SyntheticDatasetOutput:
        return self.chain.invoke({
            "schema": schema.model_dump_json(indent=2),
            "user_input": user_input,
            "num_records": num_records,
        })


# ---------------------------------------------------------------------
# 5. Integrity and fidelity checker
# ---------------------------------------------------------------------

class DataIntegrityChecker:
    """Verifies generated data programmatically and with LLM assistance."""

    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        self.llm = llm

    def check_referential_integrity(
        self, schema: RelationalSchemaOutput, data: SyntheticDatasetOutput
    ) -> Dict[str, Any]:
        """Check that every foreign key value exists in its parent table."""
        table_data = {t.table_name: t.records for t in data.dataset}
        errors: List[str] = []

        for rel in schema.relationships:
            parent_rows = table_data.get(rel.to_table)
            child_rows = table_data.get(rel.from_table)

            if parent_rows is None or child_rows is None:
                errors.append(f"Missing data for relationship {rel.from_table} -> {rel.to_table}")
                continue

            parent_keys = {row[rel.to_column] for row in parent_rows if rel.to_column in row}

            for idx, row in enumerate(child_rows):
                fk_value = row.get(rel.from_column)
                if fk_value is not None and fk_value not in parent_keys:
                    errors.append(
                        f"Orphan record in {rel.from_table}[row {idx}]: "
                        f"{rel.from_column}={fk_value} not found in {rel.to_table}.{rel.to_column}"
                    )

        return {"passed": not errors, "error_count": len(errors), "errors": errors}

    def check_statistical_fidelity(self, user_input: str, data: SyntheticDatasetOutput) -> str:
        """Ask the LLM to review realism and business-rule alignment."""
        if self.llm is None:
            return "Fidelity evaluation skipped (no LLM provided)."

        prompt = (
            f"Original input and business rules:\n{user_input}\n\n"
            f"Generated data:\n{data.model_dump_json(indent=2)}\n\n"
            "Analyze the data for statistical fidelity, realism, and alignment with the business rules. "
            "Highlight any anomalies or issues."
        )
        response = self.llm.invoke(prompt)
        return content_to_text(response.content)


# ---------------------------------------------------------------------
# 6. Pipeline orchestrator
# ---------------------------------------------------------------------

class SyntheticDataPipeline:
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL, temperature: float = 0.2):
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name, temperature=temperature)
        self.extractor = SchemaExtractor(llm)
        self.generator = SyntheticDataGenerator(llm)
        self.checker = DataIntegrityChecker(llm)

    def run(self, user_input: str, num_records: int = 5) -> Dict[str, Any]:
        print("[1/3] Extracting schema and foreign key relationships...")
        schema = self.extractor.extract(user_input)

        print("[2/3] Generating synthetic data...")
        dataset = self.generator.generate(schema, user_input, num_records)

        print("[3/3] Verifying data integrity and fidelity...")
        integrity_report = self.checker.check_referential_integrity(schema, dataset)
        fidelity_report = self.checker.check_statistical_fidelity(user_input, dataset)

        return {
            "schema": schema.model_dump(),
            "generated_data": dataset.model_dump(),
            "integrity_report": integrity_report,
            "fidelity_report": fidelity_report,
        }


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def print_section(title: str, content: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(content)


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file or environment variables.")

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    user_input = """
    We need an e-commerce database with two main tables: Users and Orders.
    - Users have user_id, full_name, email, and created_at.
    - Orders have order_id, user_id, order_date, total_amount, and status (PENDING, COMPLETED, CANCELLED).
    - Business rules:
        1. total_amount must be greater than 0.
        2. Status should predominantly be COMPLETED, but include edge cases with CANCELLED
           or negative/zero amounts flagged as anomalies.
        3. Foreign key: Orders(user_id) references Users(user_id).
    """

    print(f"Using model: {model_name}\n")
    pipeline = SyntheticDataPipeline(api_key=api_key, model_name=model_name)
    result = pipeline.run(user_input=user_input, num_records=5)

    print_section("EXTRACTED SCHEMA AND RELATIONSHIPS", json.dumps(result["schema"], indent=2))
    print_section("GENERATED DATA", json.dumps(result["generated_data"], indent=2, default=str))
    print_section("INTEGRITY REPORT", json.dumps(result["integrity_report"], indent=2))
    print_section("FIDELITY EVALUATION", result["fidelity_report"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()