import fs from "fs";
import path from "path";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { LocalBGEM3Embeddings } from "./embeddings.js";
import { config } from "./config.js";

async function buildVectorStore() {
  if (!fs.existsSync(config.dataDir)) {
    fs.mkdirSync(config.dataDir, { recursive: true });
    console.log(`Created '${config.dataDir}'. Add your English PDFs and rerun.`);
    return;
  }

  const files = fs.readdirSync(config.dataDir).filter(f => f.endsWith(".pdf"));
  if (files.length === 0) {
    console.log(`No PDF files found in '${config.dataDir}'.`);
    return;
  }

  console.log(`Loading ${files.length} PDF(s)...`);
  const rawDocs = [];
  for (const file of files) {
    const loader = new PDFLoader(path.join(config.dataDir, file));
    const docs = await loader.load();
    rawDocs.push(...docs);
  }

  console.log(`Loaded ${rawDocs.length} pages. Chunking text...`);
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap
  });
  const chunks = await splitter.splitDocuments(rawDocs);
  console.log(`Created ${chunks.length} chunks.`);

  console.log("Generating BGE-M3 embeddings & building FAISS index...");
  const embeddings = new LocalBGEM3Embeddings();
  const vectorStore = await FaissStore.fromDocuments(chunks, embeddings);

  await vectorStore.save(config.indexPath);
  console.log(`FAISS index successfully saved to '${config.indexPath}'.`);
}

buildVectorStore().catch(console.error);