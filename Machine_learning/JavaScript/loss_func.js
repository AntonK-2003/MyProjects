function loss(ysPredicted, ysReal) {
    const squaredSum = ysPredicted.reduce(
        (sum, yPredicted, i) => sum + (yPredicted - ysReal[i]) ** 2,
        0);
    return squaredSum / ysPredicted.length;
}

function loss(ysPredicted, ysReal) => {
        const ysPredictedTensor = tf.tensor(ysPredicted);
        const ysRealTensor = tf.tensor(ysReal);
        const loss = ysPredictedTensor.sub(ysRealTensor).square().mean();
        return loss.dataSync()[0];
    };

