import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createPredictionService } from "./api";
import { App } from "./App";
import "./styles.css";

const service = createPredictionService(import.meta.env);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App service={service} />
  </StrictMode>,
);
