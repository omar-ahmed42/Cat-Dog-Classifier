public class FeedForwardLayer {

    private int NEURONS_SIZE;
    private ActivationFunction activationFunction;
    private Neuron[] neurons;

    private FeedForwardLayer next;
    private FeedForwardLayer previous;

    public FeedForwardLayer(int numberOfNeurons){
        activationFunction = new SigmoidActivationFunction();
        NEURONS_SIZE = numberOfNeurons;
        neurons = new Neuron[NEURONS_SIZE];
    }

    public int getNEURONS_SIZE() {
        return NEURONS_SIZE;
    }

    public void setNEURONS_SIZE(int NEURONS_SIZE) {
        this.NEURONS_SIZE = NEURONS_SIZE;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public FeedForwardLayer getNext() {
        return next;
    }

    public void setNext(FeedForwardLayer next) {
        this.next = next;
    }

    public FeedForwardLayer getPrevious() {
        return previous;
    }

    public void setPrevious(FeedForwardLayer previous) {
        this.previous = previous;
    }

    public void setNeurons(Neuron[] neurons) {
        this.neurons = neurons;
    }

    public boolean isInput(){
        return previous == null;
    }

    public boolean isHidden(){
        return previous != null & next != null;
    }

    public boolean isOutput(){
        return next == null;
    }
}
