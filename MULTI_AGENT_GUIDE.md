# 🤖 Multi-Agent Orchestration - Quick Start

## What is Multi-Agent Orchestration?

The Multi-Agent Orchestrator combines **two intelligent agents**:

1. **🔍 RAG Agent**: Retrieves relevant context from your documents
2. **🧠 Gemini Agent**: Generates intelligent, context-aware responses

This creates a powerful Q&A system that understands Arabic perfectly!

---

## 🚀 Setup

### 1. API Key is Already Configured ✅

Your Gemini API key is stored in `.env`:
```
GEMINI_API_KEY=AIzaSyAtjMlfxSkQqq9aCcOtA1xYjf66rIvggrc
```

### 2. Install Dependencies

```bash
pip install google-genai python-dotenv
```

---

## 💡 Usage Examples

### Example 1: Simple Question

```python
from multi_agent import MultiAgentOrchestrator

# Initialize
orchestrator = MultiAgentOrchestrator()

# Ask a question
result = orchestrator.ask("ما هي خدمات إعادة التدوير؟")

print(result['answer'])
```

### Example 2: Interactive Chat

```python
from multi_agent import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator()

# Start chat interface
orchestrator.chat()
```

### Example 3: Get Context with Answer

```python
result = orchestrator.ask(
    "كيف أستفيد من إعادة التدوير؟",
    return_context=True
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['context']['num_chunks']} documents")
```

---

## 🎮 Run the Examples

### Quick Test

```bash
py test_multi_agent.py
```

### Full Demo with Chat

```bash
py multi_agent.py
```

This will:
1. Run example queries
2. Start an interactive chat session

---

## 🔧 How It Works

```
User Question
     ↓
┌────────────────┐
│  RAG Agent     │  → Searches vector database
│  (Retrieval)   │  → Finds relevant chunks
└────────────────┘
     ↓
  Context Docs
     ↓
┌────────────────┐
│ Gemini Agent   │  → Reads context + question
│ (Generation)   │  → Generates intelligent answer
└────────────────┘
     ↓
  Final Answer
```

---

## ⚙️ Configuration

Edit `.env` to customize:

```env
# Model selection
GEMINI_MODEL=gemini-2.0-flash-exp

# Number of context chunks
MAX_RESULTS=5

# Response creativity (0=deterministic, 1=creative)
TEMPERATURE=0.7
```

---

## 📊 Example Output

```
🎯 User Query: ما هي خدمات إعادة التدوير المتوفرة؟

🔍 RAG Agent: Retrieving context...
✅ Retrieved 5 relevant chunks

🧠 Gemini Agent: Generating response...
✅ Response generated

💡 Answer:
يتوفر في الأردن العديد من خدمات إعادة التدوير التي تشمل:

1. **إعادة تدوير البلاستيك**: يعمل في هذا القطاع حوالي 614 شركة ومصنع...

2. **إعادة تدوير الحديد والمعادن**: قطاع متطور يعمل على إعادة تدوير المعادن...

3. **إعادة تدوير الورق والكرتون**: يوجد حوالي 20 مصنع صغير...

📚 Context: 5 chunks
```

---

## 🎯 Advanced Usage

### Custom System Prompt

```python
custom_prompt = """أنت خبير في البيئة والاستدامة.
قدم إجابات تفصيلية مع أمثلة عملية."""

result = orchestrator.ask(
    "كيف أبدأ مشروع إعادة تدوير؟",
    system_prompt=custom_prompt
)
```

### Adjust Number of Context Chunks

```python
# Use more context for complex questions
result = orchestrator.ask(
    "ما هي التحديات والفرص في قطاع إعادة التدوير؟",
    n_results=10
)
```

---

## 🔒 Security Note

- ✅ API key is stored in `.env` (not committed to git)
- ✅ `.env` is added to `.gitignore`
- ⚠️ Never share your API key publicly

---

## 🎓 Next Steps

1. **Test with your questions**: Try different Arabic queries
2. **Explore the code**: Check `multi_agent.py` for implementation
3. **Customize prompts**: Adjust system prompts for your use case
4. **Add more documents**: Process additional files for richer context

---

**🚀 Ready to use! Start asking questions in Arabic!**
