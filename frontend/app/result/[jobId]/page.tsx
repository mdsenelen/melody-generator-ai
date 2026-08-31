import ResultView from "./result-view";

type ResultPageProps = {
  params: Promise<{ jobId: string }>;
};

export default async function ResultPage({ params }: ResultPageProps) {
  const { jobId } = await params;
  return <ResultView jobId={jobId} />;
}
