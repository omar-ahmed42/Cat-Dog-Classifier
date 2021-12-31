public class Neuron {

    private double error;
    private double output;
    private double[] weights; //weights[0]

    public Neuron(){}
    public Neuron(double[] weights) {
        this.weights = weights;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }
}
