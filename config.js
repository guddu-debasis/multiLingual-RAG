import dotenv from "dotenv";
dotenv.config();

export const config = {
  groqApiKey: process.env.GROQ_API_KEY,
  modelName: "openai/gpt-oss-120b",
  embeddingModel: "Xenova/paraphrase-multilingual-MiniLM-L12-v2", // Runs quantized BGE-M3 in Node via ONNX
  dataDir: "./data",
  indexPath: "./faiss_index",
  chunkSize: 600,
  chunkOverlap: 100,
  topK: 3
};