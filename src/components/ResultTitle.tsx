interface Props {
  predictions: {
    correct: boolean;
  }[];
}
export const ResultTitle = ({ predictions }: Props) => {
  if (predictions.length === 0) return "预测结果";

  const correctCount = predictions.filter((item) => item.correct).length;
  const totalCount = predictions.length;

  const accuracy = Math.round((correctCount / totalCount) * 100);

  return `预测结果：（${correctCount}/${totalCount} ≈ ${accuracy}%）`;
};
