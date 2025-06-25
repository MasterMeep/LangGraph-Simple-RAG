router_instructions = """You are an expert at routing a user question to a vectorstore or web search

The vector store contains documents about agents and prompt engineering

Use the vectorstore for these topics. For all else use web search

Return a JSON with a single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

doc_grader_instructions = """You are a grader assessing the relevance of a retrieved document to a user question

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant"""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

Think carefully and objectively assess whether the document contains at least some information relevant to the question.

Return a JSON with a single key, binary_score, that is 'yes' or 'no' to indicate whether the document contains at least some information relevant to the quesiton"""

rag_prompt = """You are an assistant for question answering tasks

here is the context to use to answer the users question: \n\n {context}

think carefully about the above context.

Now, review the user question: \n\n {question}

Provide an answer to this question using only the above context

Use three sentences maximum and keep the answers concise

Answer:"""

hallucination_grader_instructions = """You are a teacher grading a quiz

You will be given FACTS and a STUDENT ANSWER

Here is the criteria to follow

(1) Ensure that the STUDENT ANSWER is grounded in the FACTS

(2) Ensure that the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS

Score:

A score of yes means that the student's answer meets all of the criteria. This is the best score

A score of no mean that the student's answer fails to meet all of the criteria. This is the lowest score you can give

Explain your reasoning in a step by step manner to ensure that your reasoning and conclusion are correct

Avoid simply stating the correct answer

You are not grading how accurate the student answer is, only if it meets the above criteria, which is to ensure that the STUDENT ANSWER comes only from the FACTS

Do not grade 'no' if the STUDENT ANSWER is incorrect, only grade no if the answer does not come from the FACTS.

Grade 'yes' if the STUDENT ANSWER is fully grounded in the FACTS, regardless of factual accuracy"""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}

return a JSON with two keys, binary_score is 'yes' or 'no' score to grade whether the STUDENT ANSWER is grounded in FACTS, and a key, explanation, which contains an explanation of the score"""

final_answer_grader_instructions = """You are a teacher grading a quiz

You will be given a QUESTION and a STUDENT ANSWER

Here is the criteria to follow

(1) Ensure that the STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the best score

A score of no mean that the student's answer fails to meet all of the criteria. This is the lowest score you can give

Explain your reasoning in a step by step manner to ensure that your reasoning and conclusion are correct

Avoid simply stating the correct answer
"""

final_answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}

return a JSON with two keys, binary_score is 'yes' or 'no' score to grade whether the STUDENT ANSWER helps to answer the QUESTION and meets the criteria, and a key, explanation, which contains an explanation of the score""" 