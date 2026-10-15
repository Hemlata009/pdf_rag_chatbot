evaluation_dataset = [

    # ==========================================
    # 1. FACTUAL - COMPANY / INTERNSHIP
    # ==========================================

    {
        "question": "What was Hemlata's role at VerbX?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p4::c1"}
    },

    {
        "question": "What was the main focus of Hemlata's internship at VerbX?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p4::c1", "PBI_midterm_report.pdf::p4::c2"}
    },

    {
        "question": "What types of internal materials did Hemlata review at the beginning of the internship?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p4::c1"}
    },

    {
        "question": "What kind of communication platform is VerbX developing?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p5::c1"}
    },

    {
        "question": "Which sectors is the VerbX platform intended to serve?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p5::c1"}
    },

    # ==========================================
    # 2. OBJECTIVES / LEARNING
    # ==========================================

    {
        "question": "What were the main objectives of the internship?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p6::c1", "PBI_midterm_report.pdf::p7::c1"}
    },

    {
        "question": "What did the objective of improving data quality involve?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p7::c1"}
    },

    {
        "question": "What was the primary focus during the first week of the internship?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p8::c1"}
    },

    {
        "question": "What did the initial learning phase help Hemlata understand?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p8::c1"}
    },

    # ==========================================
    # 3. DATA ORGANIZATION / PROCESSING
    # ==========================================

    {
        "question": "Why was the internship data converted into structured Excel sheets?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p9::c1"}
    },

    {
        "question": "How was the raw data organized into a consistent format?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p9::c1"}
    },

    {
        "question": "What data cleaning activities were performed during the internship?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p10::c1"}
    },

    {
        "question": "How was useful information extracted from the datasets?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p10::c1"}
    },

    # ==========================================
    # 4. PITCH DECK / RESULTS
    # ==========================================

    {
        "question": "How did Hemlata support pitch deck preparation?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p11::c1"}
    },

    {
        "question": "What improvements resulted from organizing and cleaning the datasets?",
        "relevant_chunk_ids": {"PBI_midterm_report.pdf::p10::c1", "PBI_midterm_report.pdf::p12::c1"}
    },

    # ==========================================
    # 5. NEGATIVE / NO-ANSWER
    # ==========================================

    {
        "question": "How many KPI dashboards did Hemlata build during the internship?",
        "relevant_chunk_ids": set()
    },
]
