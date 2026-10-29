import { describe, expect, it } from "vitest";
import { parseSSE, type SSEMessage } from "./sse";

function createStreamFromChunks(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index++;
      } else {
        controller.close();
      }
    },
  });
}

describe("parseSSE", () => {
  it("parses single complete SSE message", async () => {
    const stream = createStreamFromChunks(["id: 123\ndata: hello world\n\n"]);
    const messages: SSEMessage[] = [];
    for await (const msg of parseSSE(stream)) {
      messages.push(msg);
    }

    expect(messages).toEqual([{ id: "123", event: undefined, data: "hello world" }]);
  });

  it("handles chunk boundary splits across lines and data tokens", async () => {
    const chunks = [
      ": ping\n\nid: ev",
      "t_1\nevent: mess",
      "age\ndata: {\"key\": \"va",
      "lue\"}\n\n",
    ];
    const stream = createStreamFromChunks(chunks);
    const messages: SSEMessage[] = [];
    for await (const msg of parseSSE(stream)) {
      messages.push(msg);
    }

    expect(messages).toEqual([
      { id: "evt_1", event: "message", data: '{"key": "value"}' },
    ]);
  });

  it("handles [DONE] termination marker", async () => {
    const chunks = [
      "data: chunk 1\n\n",
      "data: chunk 2\n\n",
      "data: [DONE]\n\n",
      "data: should not appear\n\n",
    ];
    const stream = createStreamFromChunks(chunks);
    const messages: SSEMessage[] = [];
    for await (const msg of parseSSE(stream)) {
      messages.push(msg);
    }

    expect(messages).toEqual([
      { id: undefined, event: undefined, data: "chunk 1" },
      { id: undefined, event: undefined, data: "chunk 2" },
      { id: undefined, event: undefined, data: "[DONE]" },
    ]);
  });

  it("joins multi-line data payloads", async () => {
    const stream = createStreamFromChunks([
      "data: line 1\ndata: line 2\ndata: line 3\n\n",
    ]);
    const messages: SSEMessage[] = [];
    for await (const msg of parseSSE(stream)) {
      messages.push(msg);
    }

    expect(messages).toEqual([
      { id: undefined, event: undefined, data: "line 1\nline 2\nline 3" },
    ]);
  });
});

