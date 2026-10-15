"""
End-to-end single-turn accuracy benchmark.

IMPORTANT:
- The original Experiment-1..4 evaluation_dataset.py is preserved.
- Precision@4, Recall@4 and MRR remain retrieval diagnostics.
- This file evaluates answer quality using gold answers and evidence.
"""

ACCURACY_DATASET = [
    {
        "id": 1,
        "question": "What was Hemlata's role at VerbX?",
        "gold_answer": (
            "Hemlata's role at VerbX was Product and Revenue "
            "Intelligence Support."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p1::c1",
            "PBI_midterm_report.pdf::p4::c1",
        ],
        "supporting_pages": [1, 4],
        "question_type": "multi_chunk_factual",
        "expected_abstention": False,
    },
    {
        "id": 2,
        "question": "What was the main focus of Hemlata's internship at VerbX?",
        "gold_answer": (
            "The internship focused mainly on organizing operational "
            "data and preparing structured information that could "
            "assist internal decision making."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p4::c1",
        ],
        "supporting_pages": [4],
        "question_type": "factual",
        "expected_abstention": False,
    },
    {
        "id": 3,
        "question": (
            "What types of internal materials did Hemlata review "
            "at the beginning of the internship?"
        ),
        "gold_answer": (
            "She reviewed internal files, spreadsheets, and documents "
            "to understand the product concept, business model, and "
            "operational workflows."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p4::c1",
        ],
        "supporting_pages": [4],
        "question_type": "factual",
        "expected_abstention": False,
    },
    {
        "id": 4,
        "question": "What kind of communication platform is VerbX developing?",
        "gold_answer": (
            "VerbX is developing a multilingual communication platform."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p5::c1",
        ],
        "supporting_pages": [5],
        "question_type": "factual",
        "expected_abstention": False,
    },
    {
        "id": 5,
        "question": "Which sectors is the VerbX platform intended to serve?",
        "gold_answer": (
            "The platform is intended to serve banking, travel, "
            "healthcare, and enterprise environments."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p5::c1",
        ],
        "supporting_pages": [5],
        "question_type": "list",
        "expected_abstention": False,
    },
    {
        "id": 6,
        "question": "What were the main objectives of the internship?",
        "gold_answer": (
            "The main objectives were understanding the system, "
            "organizing raw data, improving data quality, and "
            "supporting business documentation."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p6::c1",
            "PBI_midterm_report.pdf::p7::c1",
        ],
        "supporting_pages": [6, 7],
        "question_type": "multi_chunk_list",
        "expected_abstention": False,
    },
    {
        "id": 7,
        "question": "What did improving data quality involve?",
        "gold_answer": (
            "It involved identifying duplicate records, correcting "
            "inconsistent formatting, and organizing information "
            "into clear categories."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p7::c1",
        ],
        "supporting_pages": [7],
        "question_type": "factual",
        "expected_abstention": False,
    },
    {
        "id": 8,
        "question": "What was the primary focus during the first week?",
        "gold_answer": (
            "The primary focus was understanding the internal structure "
            "and documentation of VerbX."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p8::c1",
        ],
        "supporting_pages": [8],
        "question_type": "factual",
        "expected_abstention": False,
    },
    {
        "id": 9,
        "question": "What did the initial learning phase help Hemlata understand?",
        "gold_answer": (
            "It helped her understand the product idea and the types "
            "of operational data used within the organization."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p8::c1",
        ],
        "supporting_pages": [8],
        "question_type": "factual",
        "expected_abstention": False,
    },
    {
        "id": 10,
        "question": "Why was the internship data converted into structured Excel sheets?",
        "gold_answer": (
            "Some information was not directly suitable for analysis, "
            "so it was converted into structured Excel sheets for "
            "consistent storage and management."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p9::c1",
        ],
        "supporting_pages": [9],
        "question_type": "why",
        "expected_abstention": False,
    },
    {
        "id": 11,
        "question": "How was the raw data organized into a consistent format?",
        "gold_answer": (
            "Useful information was identified and arranged into rows "
            "and columns in structured Excel sheets."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p9::c1",
        ],
        "supporting_pages": [9],
        "question_type": "how",
        "expected_abstention": False,
    },
    {
        "id": 12,
        "question": "What data cleaning activities were performed?",
        "gold_answer": (
            "Duplicate entries were removed and formatting "
            "inconsistencies were corrected."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p10::c1",
        ],
        "supporting_pages": [10],
        "question_type": "factual",
        "expected_abstention": False,
    },
    {
        "id": 13,
        "question": "How was useful information extracted from the datasets?",
        "gold_answer": (
            "Relevant fields were identified and arranged in a way "
            "useful for internal operational purposes."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p10::c1",
        ],
        "supporting_pages": [10],
        "question_type": "how",
        "expected_abstention": False,
    },
    {
        "id": 14,
        "question": "How did Hemlata support pitch deck preparation?",
        "gold_answer": (
            "She organized relevant information into a clear and "
            "logical structure so it could be used in presentation slides."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p11::c1",
        ],
        "supporting_pages": [11],
        "question_type": "how",
        "expected_abstention": False,
    },
    {
        "id": 15,
        "question": "What improvements resulted from organizing and cleaning the datasets?",
        "gold_answer": (
            "The datasets became easier to manage, information was "
            "more accessible for operational tasks, and cleaning and "
            "organization improved data reliability."
        ),
        "supporting_chunks": [
            "PBI_midterm_report.pdf::p10::c1",
            "PBI_midterm_report.pdf::p12::c1",
        ],
        "supporting_pages": [10, 12],
        "question_type": "multi_chunk_synthesis",
        "expected_abstention": False,
    },
    {
        "id": 16,
        "question": "What was Hemlata's salary during the internship?",
        "gold_answer": "The document does not provide Hemlata's salary.",
        "supporting_chunks": [],
        "supporting_pages": [],
        "question_type": "negative",
        "expected_abstention": True,
    },
]
