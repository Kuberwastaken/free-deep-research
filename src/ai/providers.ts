import { LlamaModel, LlamaContext, LlamaChatSession } from 'node-llama-cpp';
import { getEncoding } from 'js-tiktoken';

import { RecursiveCharacterTextSplitter } from './text-splitter';

// Path to the DeepSeek model - you'll need to adjust this
const MODEL_PATH = '/path/to/deepseek-r1-1.5b-chat.gguf';

// Create a local DeepSeek model
const deepSeekModel = await createDeepSeekModel();

// Providers
async function createDeepSeekModel() {
  const model = new LlamaModel({
    modelPath: MODEL_PATH,
    contextSize: 4096,
    gpuLayers: 32, // Use GPU layers if available
  });

  const context = new LlamaContext({ model });
  return new LlamaChatSession({ context });
}

// Models - adapt to DeepSeek's capabilities
export const gpt4Model = deepSeekModel;
export const gpt4MiniModel = deepSeekModel;
export const o3MiniModel = deepSeekModel;

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// Maintain the existing trimPrompt functionality
export function trimPrompt(prompt: string, contextSize = 120_000) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  return trimPrompt(trimmedPrompt, contextSize);
}