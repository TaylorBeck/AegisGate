export interface SSEMessage {
  id?: string;
  event?: string;
  data: string;
}

/**
 * Parses a byte stream into SSE messages.
 * Handles chunk boundaries splitting across lines, SSE comments (e.g. `: ping`),
 * `id:` fields, and `[DONE]` termination markers.
 */
export async function* parseSSE(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEMessage, void, unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  let currentId: string | undefined;
  let currentEvent: string | undefined;
  let dataLines: string[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split(/\r\n|\r|\n/);
      // The last element is either empty (if ended with newline) or an incomplete line
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line === "") {
          // Empty line indicates event dispatch
          if (dataLines.length > 0) {
            const data = dataLines.join("\n");
            yield {
              id: currentId,
              event: currentEvent,
              data,
            };
            if (data === "[DONE]") {
              return;
            }
            dataLines = [];
            currentEvent = undefined;
          }
          continue;
        }

        if (line.startsWith(":")) {
          // Comment line, ignore (e.g. ": ping")
          continue;
        }

        if (line.startsWith("id:")) {
          currentId = line.slice(3).trimStart();
        } else if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trimStart();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
    }

    // Flush any remaining buffer if it contains complete content
    if (buffer.length > 0) {
      if (buffer.startsWith("data:")) {
        dataLines.push(buffer.slice(5).trimStart());
      }
      if (dataLines.length > 0) {
        yield {
          id: currentId,
          event: currentEvent,
          data: dataLines.join("\n"),
        };
      }
    }
  } finally {
    reader.releaseLock();
  }
}

