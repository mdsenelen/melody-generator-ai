import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { UploadButton } from "../../components/upload-button";
import { uploadFile } from "../../app/lib/upload";

jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));

const mockedUploadFile = uploadFile as jest.Mock;

describe("UploadButton", () => {
  beforeEach(() => {
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "clip.wav" });
  });

  afterEach(() => {
    jest.clearAllMocks();
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
