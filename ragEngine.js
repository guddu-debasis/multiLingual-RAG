import Groq from "groq-sdk";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { LocalBGEM3Embeddings } from "./embeddings.js";
import { config } from "./config.js";

const groq = new Groq({ apiKey: config.groqApiKey });

export async function createRAGEngine() {
  const embeddings = new LocalBGEM3Embeddings();
  const vectorStore = await FaissStore.load(config.indexPath, embeddings);

  return {
    async ask(query) {
      // 1. Cross-Lingual Search (Hindi query -> English docs)
      const relevantDocs = await vectorStore.similaritySearch(query, config.topK);
      const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

      // 2. Multilingual System Prompt
      const systemPrompt = `You are a helpful multilingual assistant.
Context (from English documents):
${context}

Instructions:
1. Answer the question accurately using ONLY the provided context.
2. If the user asks in Hindi (Devanagari or Hinglish), answer in natural Hindi.
3. If the context does not contain the answer, politely state that you do not know in the user's language.`;

      // 3. Groq API Call
      const chatCompletion = await groq.chat.completions.create({
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: query }
        ],
        model: config.modelName,
        temperature: 0.1
      });

      return {
        answer: chatCompletion.choices[0]?.message?.content || "",
        sources: relevantDocs
      };
    }
  };
}