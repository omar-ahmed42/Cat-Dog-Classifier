public class SigmoidActivationFunction implements ActivationFunction{

    @Override
    public double activateActivationFunction(double x) {
        return 1/(1 + Math.exp(-x));
    }
}
