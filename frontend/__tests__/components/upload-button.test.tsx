import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { UploadButton } from "../../components/upload-button";
import { uploadFile } from "../../app/lib/upload";

jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));

const mockedUploadFile = uploadFile as jest.Mock;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("UploadButton", () => {
  beforeEach(() => {
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "clip.wav" });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("keeps only the animated uploading status visible while uploading", async () => {
    const pendingUpload = deferred<{ id: string; filename: string }>();
    mockedUploadFile.mockReturnValue(pendingUpload.promise);
    const user = userEvent.setup();
    render(<UploadButton onUploadSuccess={jest.fn()} />);

    await user.upload(screen.getByLabelText("Upload audio file"), new File(["audio"], "clip.wav"));

    expect(screen.queryByText("Uploading...")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Uploading");

    pendingUpload.resolve({ id: "up-1", filename: "clip.wav" });
  });

  it("shows upload completion only briefly after a successful upload", async () => {
    const user = userEvent.setup();
    render(<UploadButton onUploadSuccess={jest.fn()} />);

    expect(screen.queryByText("Upload complete")).not.toBeInTheDocument();

    await user.upload(screen.getByLabelText("Upload audio file"), new File(["audio"], "clip.wav"));

    expect(screen.getByText("Upload complete")).toBeInTheDocument();

    await waitFor(() => expect(screen.queryByText("Upload complete")).not.toBeInTheDocument(), {
      timeout: 3000,
    });
  });
});
