
public class Temperature {

    /**
     * Converts kelvin to celsius.
     * @param tempInCelsius temperature in kelvin
     * @return temperature in celsius
     */
    public double toCelsius(double tempInKelvin) {
        return tempInKelvin - 273.15;
    }

    /**
     * Returns the maximum of the two values.
     * @param a first value
     * @param b second value
     * @return the smaller of a and b
     */
    public int pick(int a, int b) {
        return Math.min(a, b);
    }

    /**
     * Reads a file.
     * @param path the path
     * @return the content
     */
    public String read(String path) throws java.io.IOException {
        return helper(path);
    }

    private String helper(String p) { return p; }
}