import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import sampleCsv from "../public/samples/hotels_reviews_sample.csv?raw";
import { App } from "./App";
import { PredictionError, type BatchPredictRequest, type PredictionService } from "./api";
import { createMockPredictionService } from "./api/mockService";

afterEach(() => {
  vi.restoreAllMocks();
});

function file(name: string, content: string) {
  return new File([content], name, { type: "text/plain" });
}

async function openBatchTab() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("tab", { name: "Review file & dashboard" }));
  return user;
}

async function uploadAndAnalyze(user: ReturnType<typeof userEvent.setup>, upload: File) {
  await user.upload(screen.getByLabelText("Reviews file"), upload);
  await screen.findByTestId("file-summary");
  await user.click(screen.getByRole("button", { name: "Analyze file" }));
}

function aspectRow(name: string) {
  return screen.getByRole("button", { name }).closest("tr")!;
}

describe("tabs", () => {
  it("switches with click and arrow keys and keeps each tab's state", async () => {
    render(<App service={createMockPredictionService({ delayMs: 0 })} />);
    const user = userEvent.setup();
    const single = screen.getByRole("tab", { name: "Single review" });
    expect(single).toHaveAttribute("aria-selected", "true");

    await user.type(screen.getByLabelText("Sentence"), "Kept text");
    single.focus();
    await user.keyboard("{ArrowRight}");
    const batch = screen.getByRole("tab", { name: "Review file & dashboard" });
    expect(batch).toHaveAttribute("aria-selected", "true");
    expect(batch).toHaveFocus();
    expect(screen.getByLabelText("Reviews file")).toBeVisible();

    await user.keyboard("{ArrowLeft}");
    expect(screen.getByLabelText("Sentence")).toHaveValue("Kept text");
  });
});

describe("review file dashboard", () => {
  it("analyzes the sample file in mock mode and highlights the most negative aspect", async () => {
    render(<App service={createMockPredictionService({ delayMs: 0 })} />);
    const user = await openBatchTab();

    await uploadAndAnalyze(user, file("hotels_reviews_sample.csv", sampleCsv));

    const weakest = await screen.findByTestId("weakest-aspect");
    expect(within(weakest).getByRole("heading")).toHaveTextContent("Most negative aspect: rooms cleanliness");
    expect(weakest).toHaveTextContent("74% of the 19 reviews that mention it are negative");
    expect(weakest).toHaveTextContent("“dirty” (7)");
    expect(screen.getByText(/keyword rules, not the ASQP model/)).toBeInTheDocument();

    const row = aspectRow("rooms cleanliness");
    expect(within(row).getAllByRole("cell").map((cell) => cell.textContent)).toEqual([
      "19",
      "26% (5)",
      "74% (14)",
      "0% (0)",
      "0% (0)",
      "",
    ]);
    // Default sort is % negative among aspects with at least 3 reviews.
    const aspects = within(screen.getByRole("table")).getAllByRole("rowheader").map((cell) => cell.textContent);
    expect(aspects[0]).toBe("rooms cleanliness");
    expect(aspects).not.toContain("hotel prices");
    expect(screen.getByText(/aspect\(s\) with fewer than 3 reviews are hidden/)).toBeInTheDocument();
  });

  it("drills into an aspect to show complaints and matching reviews", async () => {
    render(<App service={createMockPredictionService({ delayMs: 0 })} />);
    const user = await openBatchTab();
    await uploadAndAnalyze(user, file("sample.csv", sampleCsv));

    await user.click(await screen.findByRole("button", { name: "See reviews for rooms cleanliness" }));

    const detail = screen.getByRole("region", { name: "rooms cleanliness" });
    expect(within(detail).getByRole("heading", { name: "rooms cleanliness" })).toHaveFocus();
    expect(within(detail).getByText("Top negative opinions").nextElementSibling).toHaveTextContent("“dirty” — 7 review(s)");
    const reviews = within(detail).getByRole("list", { name: "Reviews about rooms cleanliness" });
    expect(within(reviews).getAllByRole("listitem")).toHaveLength(10);
    expect(within(detail).getByText("Showing 10 of 14 reviews.")).toBeInTheDocument();

    await user.selectOptions(within(detail).getByLabelText("Show reviews"), "positive");
    expect(within(reviews).getAllByRole("listitem")).toHaveLength(5);
  });

  it("regroups by entity and applies the minimum-reviews filter", async () => {
    render(<App service={createMockPredictionService({ delayMs: 0 })} />);
    const user = await openBatchTab();
    await uploadAndAnalyze(user, file("sample.csv", sampleCsv));
    await screen.findByTestId("weakest-aspect");

    await user.selectOptions(screen.getByLabelText("Group by"), "entity");
    expect(within(aspectRow("rooms")).getAllByRole("cell")[0]).toHaveTextContent("21");

    const min = screen.getByLabelText("Minimum reviews");
    await user.clear(min);
    await user.type(min, "10");
    const aspects = within(screen.getByRole("table")).getAllByRole("rowheader").map((cell) => cell.textContent);
    expect(aspects).toEqual(["rooms", "service"]);
  });

  it("exports the visible table as CSV", async () => {
    const createObjectURL = vi.fn<(blob: Blob) => string>(() => "blob:test");
    Object.assign(URL, { createObjectURL, revokeObjectURL: vi.fn() });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    render(<App service={createMockPredictionService({ delayMs: 0 })} />);
    const user = await openBatchTab();
    await uploadAndAnalyze(user, file("sample.csv", sampleCsv));
    await screen.findByTestId("weakest-aspect");

    await user.click(screen.getByRole("button", { name: "Export CSV" }));

    expect(click).toHaveBeenCalledTimes(1);
    const csv = await createObjectURL.mock.calls[0]![0].text();
    expect(csv.split("\r\n")[1]).toMatch(/^rooms cleanliness,19,5,14,0,0,26\.3,73\.7,0\.0,0\.0,dirty \(7\)/);
  });

  it("uses predictions already in the file without calling the service", async () => {
    const service = createMockPredictionService({ delayMs: 0 });
    const spy = vi.spyOn(service, "predictBatch");
    render(<App service={service} />);
    const user = await openBatchTab();
    const lines = [1, 2, 3].map((index) =>
      JSON.stringify({
        review_id: `r${index}`,
        text: "Room was dirty.",
        quadruples: [{ target: "Room", aspect: "rooms cleanliness", opinion: "dirty", sentiment: "negative" }],
      }),
    );

    await uploadAndAnalyze(user, file("predictions.jsonl", lines.join("\n")));

    expect(await screen.findByText(/Using the predictions included in the uploaded file/)).toBeInTheDocument();
    expect(screen.getByTestId("weakest-aspect")).toHaveTextContent("100% of the 3 reviews");
    expect(spy).not.toHaveBeenCalled();
  });

  it("reports invalid files without sending anything", async () => {
    const service = createMockPredictionService({ delayMs: 0 });
    const spy = vi.spyOn(service, "predictBatch");
    render(<App service={service} />);
    const user = await openBatchTab();

    await user.upload(screen.getByLabelText("Reviews file"), file("reviews.csv", "id,body\n1,hello"));

    expect(await screen.findByRole("alert")).toHaveTextContent(/text column/);
    expect(screen.getByLabelText("Reviews file")).toHaveAttribute("aria-invalid", "true");
    await user.click(screen.getByRole("button", { name: "Analyze file" }));
    expect(spy).not.toHaveBeenCalled();
  });

  it("shows an error when the model is not ready and resumes where it stopped", async () => {
    let fail = true;
    const predictBatch = vi.fn(async (request: BatchPredictRequest) => {
      if (fail) {
        fail = false;
        throw new PredictionError("server", "Checkpoint cho domain/model này chưa sẵn sàng.", { code: "MODEL_NOT_READY" });
      }
      return { results: request.items.map((item) => ({ id: item.id, quadruples: [] })) };
    });
    const service: PredictionService = { mode: "api", predict: vi.fn(), predictBatch };
    render(<App service={service} />);
    const user = await openBatchTab();
    await user.selectOptions(
      within(screen.getByRole("tabpanel", { name: "Review file & dashboard" })).getByLabelText("Domain"),
      "coursera",
    );
    await uploadAndAnalyze(user, file("reviews.txt", "Good course.\nBoring lectures."));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Checkpoint cho domain/model này chưa sẵn sàng.");
    expect(alert).toHaveTextContent("Error code: MODEL_NOT_READY");
    expect(predictBatch.mock.calls[0]![0]).toMatchObject({ domain: "coursera", model_type: "paraphrase" });

    await user.click(within(alert).getByRole("button", { name: "Resume" }));
    expect(await screen.findByRole("heading", { name: "No aspects found" })).toBeInTheDocument();
    expect(screen.getByText(/Estimated by Paraphrase-T5 on Coursera/)).toBeInTheDocument();
  });
});
