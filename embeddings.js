import { pipeline } from "@xenova/transformers";
import { Embeddings } from "@langchain/core/embeddings";
import { config } from "./config.js";

export class LocalBGEM3Embeddings extends Embeddings {
  constructor() {
    super({});
    this.pipe = null;
  }

  async init() {
    if (!this.pipe) {
      this.pipe = await pipeline("feature-extraction", config.embeddingModel);
    }
  }

  async embedDocuments(texts) {
    await this.init();
    const embeddings = [];
    for (const text of texts) {
      const output = await this.pipe(text, { pooling: "cls", normalize: true });
      embeddings.push(Array.from(output.data));
    }
    return embeddings;
  }

  async embedQuery(text) {
    await this.init();
    const output = await this.pipe(text, { pooling: "cls", normalize: true });
    return Array.from(output.data);
  }
}