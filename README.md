# apt-rag

아파트 관리규약을 읽어 하이브리드(벡터 + BM25) 검색용 색인을 만들고, 검색된 조항을 근거로 LLM 답변(ask)을 내는 도구.

```bash
uv sync
uv run apt-rag index <pdf>
uv run apt-rag ask "<질문>"
```
