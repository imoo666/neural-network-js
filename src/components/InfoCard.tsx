import { Card } from "antd";

interface Props {
  title: string;
  accuracy: number;
  rounds: number;
  totalTime: number;
  className?: string;
}
export const InfoCard = ({
  title,
  rounds,
  totalTime,
  accuracy,
  className,
}: Props) => {
  return (
    <Card className={className} title="模型信息">
      <div>模型主题：{title}</div>
      <div>训练轮数：{rounds}轮</div>
      <div>最终准确率：{accuracy}%</div>
      <div>训练消耗时间：{totalTime}分钟（以 mac m3 芯片为例）</div>
      <br></br>
      <div>
        备注1：当观察到连续数轮训练的损失停止不动时，需要刷新页面重新训练
      </div>
      <div>
        备注2：当观察到所有训练结果均快速为 0 时，需要完全关闭浏览器并重启浏览器
      </div>
    </Card>
  );
};
