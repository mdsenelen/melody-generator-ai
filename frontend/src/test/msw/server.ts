// One MSW server for every jsdom test (started in jest.setup.js). There are
// no default handlers: each test declares the responses it depends on with
// server.use(...), and any request nobody declared fails the test.
import { setupServer } from "msw/node";

export const server = setupServer();

// Where the browser talks to FastAPI directly (uploads, job polling), as
// opposed to the same-origin Next.js /api/* proxy routes.
export const BACKEND_API = "http://127.0.0.1:8000/api";
