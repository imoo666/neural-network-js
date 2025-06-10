import * as tf from "@tensorflow/tfjs";
import { Button, Card, List, Tag } from "antd";
import { useState } from "react";
import { InfoCard } from "../../components/InfoCard";
import { ResultTitle } from "../../components/ResultTitle";

// 类型定义
interface TrainingLog {
  epoch: number;
  loss: number;
  accuracy: number;
}

interface Prediction {
  input: string;
  actual: string;
  predicted: string;
  correct: boolean;
}

// 生成规则序列
const generateBinaryString = (length: number) => {
  let result = "";
  result += Math.random() < 0.5 ? "0" : "1";
  result += Math.random() < 0.5 ? "0" : "1";

  for (let i = 2; i < length; i++) {
    const prev1 = result[i - 2];
    const prev2 = result[i - 1];
    if (prev1 === "0" && prev2 === "0") {
      result += "1";
    } else if (prev1 === "1" && prev2 === "1") {
      result += "0";
    } else {
      result += Math.random() < 0.5 ? "0" : "1";
    }
  }

  return result;
};

export const BinaryRnnPage = () => {
  const [modelState, setModelState] = useState<{
    model: tf.LayersModel | null;
    isTraining: boolean;
    logs: TrainingLog[];
  }>({ model: null, isTraining: false, logs: [] });

  const [predictions, setPredictions] = useState<Prediction[]>([]);

  // 加载数据：整个序列作为输入
  const loadSequenceData = async () => {
    const text = generateBinaryString(10000);
    const seq = text.split("").map(Number);

    const input = tf.tensor(seq.slice(0, -1)).reshape([1, seq.length - 1, 1]);
    const label = tf.tensor(seq.slice(1)).reshape([1, seq.length - 1, 1]);

    return { x: input, y: label, raw: text };
  };

  // 定义模型：RNN + Dense
  const defineModel = () => {
    const model = tf.sequential();
    model.add(
      tf.layers.simpleRNN({
        units: 32,
        inputShape: [null, 1],
        returnSequences: true,
        activation: "tanh",
      })
    );
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
      optimizer: "adam",
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    return model;
  };

  // 训练模型
  const trainModel = async () => {
    setModelState({ model: null, isTraining: true, logs: [] });

    const model = defineModel();
    const { x, y, raw } = await loadSequenceData();

    await model.fit(x, y, {
      epochs: 10,
      batchSize: 1,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (!logs) return;
          setModelState((prev) => ({
            ...prev,
            logs: [
              ...prev.logs,
              {
                epoch: epoch + 1,
                loss: Number(logs.loss?.toFixed(4)),
                accuracy: Number((logs.acc ?? logs.accuracy ?? 0).toFixed(4)),
              },
            ],
          }));
        },
      },
    });

    predict(model, raw);
    setModelState((prev) => ({ ...prev, model, isTraining: false }));
    tf.dispose([x, y]);
  };

  // 执行预测
  const predict = (model: tf.LayersModel, fullSeq: string) => {
    const input = tf
      .tensor(fullSeq.slice(0, -1).split("").map(Number))
      .reshape([1, fullSeq.length - 1, 1]);

    const output = model.predict(input) as tf.Tensor;
    const predictedVals = output.squeeze().arraySync() as number[];

    const results: Prediction[] = [];

    for (let i = 0; i < 50; i++) {
      const pred = predictedVals[i] > 0.5 ? "1" : "0";
      const actual = fullSeq[i + 1];
      results.push({
        input: fullSeq[i],
        predicted: pred,
        actual,
        correct: pred === actual,
      });
    }

    tf.dispose([input, output]);
    setPredictions(results);
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6">
      <div className="flex gap-4 items-center mb-8">
        <Button
          type="primary"
          size="large"
          onClick={trainModel}
          disabled={modelState.isTraining}
        >
          {modelState.model ? "重新训练模型" : "开始训练"}
        </Button>
        {modelState.model && <Tag color="blue">模型已就绪</Tag>}
      </div>

      <InfoCard
        className="!mb-6"
        title="RNN 序列预测"
        accuracy={
          predictions.length === 0
            ? 0
            : Math.round(
                (predictions.filter((p) => p.correct).length /
                  predictions.length) *
                  100
              )
        }
        rounds={modelState.logs.length}
        totalTime={2}
      />

      <Card title="训练日志" className="!mb-6 max-h-[300px] overflow-y-auto">
        {modelState.logs.length > 0 ? (
          <List
            bordered
            dataSource={modelState.logs}
            renderItem={(log) => (
              <List.Item>
                <div className="flex justify-between w-full">
                  <div>轮数 {log.epoch}</div>
                  <div>损失: {log.loss}</div>
                  <div>准确率: {(log.accuracy * 100).toFixed(1)}%</div>
                </div>
              </List.Item>
            )}
          />
        ) : (
          <div className="p-4 text-gray-500">
            {modelState.isTraining ? "训练进行中..." : "训练尚未开始"}
          </div>
        )}
      </Card>

      <Card title={<ResultTitle predictions={predictions} />}>
        {predictions.length > 0 ? (
          <List
            bordered
            dataSource={predictions}
            renderItem={(item) => (
              <List.Item style={{ color: item.correct ? "green" : "red" }}>
                <div className="flex gap-4">
                  <div>输入: {item.input}</div>
                  <div>预测: {item.predicted}</div>
                  <div>实际: {item.actual}</div>
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
