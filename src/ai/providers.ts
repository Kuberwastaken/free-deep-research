import { AutoModelForCausalLM, AutoTokenizer } from '@xenova/transformers';
import { getEncoding } from 'js-tiktoken';
import { RecursiveCharacterTextSplitter } from './text-splitter';

export function generateObject() {
  // Implementation of generateObject function
}

const MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B';

async function createDeepSeekModel() {
  console.log("Downloading model... This may take a while.");
  
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_NAME);
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);

  console.log("Model downloaded successfully!");
  return { model, tokenizer };
}

// Export models as promises
export const gpt4Model = createDeepSeekModel();
export const gpt4MiniModel = createDeepSeekModel();
export const o3MiniModel = createDeepSeekModel();
export const deepSeekModel = await createDeepSeekModel();

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// Function to trim prompts
export function trimPrompt(prompt: string, contextSize = 120_000): string {
  if (!prompt) return '';

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) return prompt;

  const overflowTokens = length - contextSize;
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) return prompt.slice(0, MinChunkSize);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });

  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';
  return trimmedPrompt.length === prompt.length
    ? trimPrompt(prompt.slice(0, chunkSize), contextSize)
    : trimPrompt(trimmedPrompt, contextSize);
}
