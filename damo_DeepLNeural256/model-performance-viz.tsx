import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ModelPerformanceChart = () => {
  // This is dummy data. Replace with actual training history.
  const data = Array.from({length: 50}, (_, i) => ({
    epoch: i + 1,
    loss: Math.exp(-0.05 * i) + Math.random() * 0.1,
    val_loss: Math.exp(-0.04 * i) + Math.random() * 0.15,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }} />
        <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Training Loss" />
        <Line type="monotone" dataKey="val_loss" stroke="#82ca9d" name="Validation Loss" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ModelPerformanceChart;
