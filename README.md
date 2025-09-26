# Intern_Chatbot_Thesis_RAG
In this assignment I have developed a chatbot using Retrieval Augmented Generation (RAG) that answers questions related to my thesis. RAG is a technique that can be used to increase the correctness of answers in a specific domain. Moreover, it also decreases hallucinations. The RAG pipeline can be divided in two main steps. 

The first step of the RAG pipeline is indexing. Indexing is done through splitting the input, in the form of text, in a pre-determined number of subsets called chuncks. The splitting of the text is done through the TokenTextSplitter. Alternatives for this are:
(i) RecursiveCharacterTextSplitter
(ii) NLTKTextSplitter
(iii) SpacyTextSplitter

(i) has the advantages that it bounds naturally (it respects paragraphs, sentences, and words), preserves semantics, 

Text Splitter Comparison
1. TokenTextSplitter (Currently Used)
Advantages:
✅ Token Accuracy: Ensures chunks fit precisely within LLM token limits
✅ Model Compatibility: Uses same tokenizer as embedding/LLM models
✅ Mathematical Content: Excellent for formulas, equations, and technical text
✅ Predictable Size: Consistent chunk sizes based on actual token count
✅ OpenAI Integration: Perfect alignment with OpenAI's tokenizer (o200k_base)
Disadvantages:
❌ Semantic Breaking: May split mid-sentence or mid-concept
❌ Context Loss: Can break natural language boundaries
❌ Tokenizer Dependency: Tied to specific tokenizer implementation
❌ Less Human-Readable: Chunks may not align with natural text structure
2. RecursiveCharacterTextSplitter
Advantages:
✅ Natural Boundaries: Respects paragraphs, sentences, and words
✅ Semantic Preservation: Maintains text structure and meaning
✅ Flexible Separators: Customizable hierarchy (paragraphs → sentences → words)
✅ Human-Readable: Chunks align with natural text flow
✅ Language Agnostic: Works with any text without tokenizer dependency
Disadvantages:
❌ Size Inconsistency: Chunk sizes vary significantly
❌ Token Limit Risk: May exceed LLM token limits unpredictably
❌ Mathematical Content: Can break formulas and equations awkwardly
❌ Less Precise: Character count doesn't match token count
3. NLTKTextSplitter
Advantages:
✅ Linguistic Intelligence: Uses NLTK's sentence tokenization
✅ Sentence Boundaries: Respects proper sentence structure
✅ Language Support: Handles multiple languages with NLTK
✅ Academic Text: Good for research papers and formal documents
✅ Mature Library: Well-established NLP toolkit
Disadvantages:
❌ Dependency Heavy: Requires NLTK installation and data
❌ Performance: Slower than character-based splitters
❌ Mathematical Content: Struggles with equations and formulas
❌ Size Control: Less precise control over chunk sizes
❌ Setup Complexity: Requires downloading NLTK data
4. SpacyTextSplitter
Advantages:
✅ Advanced NLP: Uses spaCy's sophisticated sentence segmentation
✅ High Accuracy: Better sentence boundary detection than NLTK
✅ Language Models: Supports many languages with trained models
✅ Academic Quality: Excellent for research and technical documents
✅ Entity Awareness: Can preserve named entities and technical terms
Disadvantages:
❌ Heavy Dependencies: Large model downloads required
❌ Memory Usage: High RAM requirements for language models
❌ Slow Processing: Significantly slower than other options
❌ Mathematical Content: Still struggles with equations and formulas
❌ Setup Complexity: Requires downloading language models
❌ Overkill: May be excessive for simple text splitting needs
Rewrite this into something more clear. 



It is ensured that there is also is some amount of overlap present to make sure the context is not lost. For example, if the text involves a mathematical theorem and proof, we want some overlap such that it is clear that these belong together. For each page in the document provided as input to the model a window of tokens is selected, with overlap, to be first summarized through a call to the OpenAI API and consequently turned into embeddings. Embeddings are vector representations of words in a continuous vector space. Asynchronous to this and in a similar fashion keywords are found of each chunk are added as metadata. The full list of the extracted metadata is:
- source 
- page number
- summary (chunk level)
- chunk identifier
- chunk index
- when a chunk is created
- key words of a chunk extracted using AI
- complete text content of a chunk 
- unique identifier for the entire document
- hash of the metadata for change detection 

The second step of the RAG pipeline is retrieval and generation. For each user question the model retrieves data, this is done in a hybrid way. Azure uses a combination of semantics and keywords. For the semantics the query embedding is compared to the chunk embeddings, based on the vector similarity metric cosine a ranking is made. In parallel, keyword search is performed through classic BM25 lexical matching. Both results are combined using Reciprocal Rank Fusion (RRF). Then re-ranking is done through Cohere. The Cohere reranker merges each candidate chunk and the query together as a single input, then a relevance score is assigned using a transformer. This reads the full token interactions between query and chunk, giving higher precision on the final ordering. The top candidates are then selected and converted into a compact "Context" string. In this way, once a query is submitted to the large language model (LLM), the answer is grounded. Grounded means that the answer must come from the provided context only, no prior knowledge is included. If the the context is missing, the model will respond with something along the lines of "I don't know". 

Furthermore, this chatbot has the following additional features: 
- Vanilla UI to ask questions and view clickable source pages from the thesis. The thesis is accessible through Azure Blob Storage. 
- Temperature of the Chat-GPT 4o-mini model is set to 0.1 such that answers are precise and concise. 
- FastAPI backend uses LangChain with Azure OpenAI to answer questions using retrieved context and trace responses.
- Streaming of the responses is done via SSE.

