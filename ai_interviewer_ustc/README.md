# ai_interviewer_ustc

## RAG技术细节

RAG，即检索增强生成，是指在通过给定一定的知识库来增强重写提示词从而获得更好的LLM表现效果的框架或技术，能够有效减少幻觉，增强模型生成的准确性和专业性。

在本次ai_interviewer中，我们使用了LangChain以及ChromaDB. 

### LangChain

LangChain能够提供统一的拼装框架：加载器、文本分割、Embedding等等，具有模块化、可插拔、样板代码少等优点。

### ChromaDB

ChromaDB是轻量级的向量数据库，支持本地、持久化的存储、metadata过滤等，以及在RAG中尤为重要的相似度最大内积检索。在RAG中，ChromaDB用来存放切分片段的向量。检索阶段相似度并返回关键片段。