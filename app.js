import readline from "readline";
import fs from "fs";
import { config } from "./config.js";
import { createRAGEngine } from "./ragengine.js";

async function main() {
  if (!fs.existsSync(config.indexPath)) {
    console.error(`Index '${config.indexPath}' not found. Run 'npm run ingest' first.`);
    process.exit(1);
  }

  console.log("Loading Multilingual Cross-Lingual RAG...");
  const engine = await createRAGEngine();
  console.log("\n--- Ready! Ask in Hindi or English (Type 'exit' to quit) ---\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const promptUser = () => {
    rl.question("User Question: ", async (query) => {
      const trimmed = query.trim();
      if (["exit", "quit", "q"].includes(trimmed.toLowerCase())) {
        rl.close();
        return;
      }
      if (!trimmed) {
        promptUser();
        return;
      }

      console.log("\nSearching & Generating response via Groq...");
      try {
        const { answer, sources } = await engine.ask(trimmed);
        
        console.log("\n--- Retrieved Passages ---");
        sources.forEach((s, idx) => {
          console.log(`[${idx + 1}] ${s.pageContent.slice(0, 90).replace(/\n/g, " ")}...`);
        });

        console.log(`\nResponse:\n${answer}\n------------------------------------------\n`);
      } catch (err) {
        console.error("Error generating answer:", err);
      }

      promptUser();
    });
  };

  promptUser();
}

main();