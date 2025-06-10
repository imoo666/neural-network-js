import * as tf from "@tensorflow/tfjs";
import { Button, Card, List } from "antd";
import { useState } from "react";
import { InfoCard } from "../../components/InfoCard";
import { ResultTitle } from "../../components/ResultTitle";

interface IrisData {
  features: number[];
  label: "setosa" | "versicolor" | "virginica";
}
interface PredictionResult {
  label: string;
  confidence: number;
  correct: boolean;
  features: number[];
  actual: string;
}

export const IrisPage = () => {
  const [modelState, setModelState] = useState<{
    model: tf.Sequential | null;
    isTraining: boolean;
    logs: { epoch: number; loss: number; accuracy: number }[];
  }>({
    model: null,
    isTraining: false,
    logs: [],
  });

  const [predictions, setPredictions] = useState<PredictionResult[]>([]);

  // 1. 加载和准备数据
  const loadAndPrepareData = async () => {
    const response = await fetch("src/pages/iris/assets/iris.txt");
    const text = await response.text();
    const data: IrisData[] = text
      .split(/\r?\n/)
      .filter((line) => line.trim())
      .map((line) => {
        const [a, b, c, d, label] = line.split(",");
        return {
          features: [a, b, c, d].map(Number),
          label: label.trim() as IrisData["label"],
        };
      });

    tf.util.shuffle(data);

    const xs = tf.tensor2d(data.slice(0, 130).map((d) => d.features));
    const ys = tf.oneHot(
      tf.tensor1d(
        data
          .slice(0, 130)
          .map((d) => ["setosa", "versicolor", "virginica"].indexOf(d.label)),
        "int32"
      ),
      3
    );

    return { xs, ys, testSamples: data.slice(-20) };
  };

  // 2. 定义模型
  const defineModel = () => {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ units: 6, activation: "relu", inputShape: [4] }),
        tf.layers.dense({ units: 6, activation: "relu" }),
        tf.layers.dense({ units: 6, activation: "relu" }),
        tf.layers.dense({ units: 3, activation: "softmax" }),
      ],
    });

    model.compile({
      optimizer: tf.train.adam(0.05),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    return model;
  };

  // 3. 训练模型
  const trainModel = async () => {
    setModelState({ model: null, isTraining: true, logs: [] });
    const model = defineModel();

    const { xs, ys, testSamples } = await loadAndPrepareData();

    await model.fit(xs, ys, {
      epochs: 20,
      batchSize: 8,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (!logs) return;
          setModelState((prev) => ({
            ...prev,
            logs: [
              ...prev.logs,
              {
                epoch: epoch + 1,
                loss: Number(logs.loss?.toFixed(4)) || 0,
                accuracy: Number(logs.acc?.toFixed(4)) || 0,
              },
            ],
          }));
        },
      },
    });

    // 训练完成后用模型预测10个测试样本
    predictSamples(model, testSamples);

    setModelState((prev) => ({ ...prev, model, isTraining: false }));
  };

  // 4. 使用模型预测多个样本
  const predictSamples = (model: tf.Sequential, samples: IrisData[]) => {
    const labels = ["setosa", "versicolor", "virginica"];

    const predictions = samples.map((sample) => {
      const input = tf.tensor2d([sample.features]);
      const output = model.predict(input) as tf.Tensor;
      const probs = Array.from(output.dataSync());
      const index = output.argMax(1).dataSync()[0];
      tf.dispose(input);
      tf.dispose(output);

      return {
        label: labels[index],
        confidence: Math.round(probs[index] * 100),
        correct: labels[index] === sample.label,
        features: sample.features,
        actual: sample.label,
      };
    });

    setPredictions(predictions);
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6">
      <div className="flex flex-wrap gap-4 items-center mb-8">
        <Button
          type="primary"
          size="large"
          onClick={trainModel}
          disabled={modelState.isTraining}
        >
          {modelState.model ? "重新训练模型" : "开始训练"}
        </Button>

        {modelState.model && (
          <span className="text-green-600 font-semibold">模型已就绪</span>
        )}
      </div>

      <InfoCard
        className="!mb-6"
        title="鸢尾花"
        accuracy={95}
        rounds={20}
        totalTime={0.1}
      />
      <Card title="训练进度" className="!mb-6">
        <div className="max-h-[300px] overflow-y-auto">
          {modelState.logs.length > 0 ? (
            <List
              bordered
              dataSource={modelState.logs}
              renderItem={(log) => (
                <List.Item>
                  <div className="flex justify-between w-full gap-4">
                    <div className="flex-1">轮数 {log.epoch}</div>
                    <div className="flex-1">损失: {log.loss}</div>
                    <div className="flex-1">
                      准确率: {(log.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                </List.Item>
              )}
            />
          ) : (
            <div className="p-4 text-gray-500">
              {modelState.isTraining ? "训练进行中..." : "训练尚未开始"}
            </div>
          )}
        </div>
      </Card>

      <Card title={<ResultTitle predictions={predictions} />}>
        {modelState.model && predictions.length > 0 ? (
          <List
            bordered
            dataSource={predictions}
            renderItem={(item) => (
              <List.Item style={{ color: item.correct ? "green" : "red" }}>
                <div className="flex justify-between w-full gap-4">
                  <div className="flex-1">数据: {item.features.join(", ")}</div>
                  <div className="flex-1">实际: {item.actual}</div>
                  <div className="flex-1">预测: {item.label}</div>
                  <div className="flex-1">置信度: {item.confidence}%</div>
                </div>
              </List.Item>
            )}
          />
        ) : (
          <div className="p-4 text-gray-500">请先训练模型</div>
        )}
      </Card>
    </div>
  );
};
