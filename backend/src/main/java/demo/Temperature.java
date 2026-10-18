package demo;

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
     * @return the larger of a and b
     */
    public int pick(int a, int b) {
        return Math.min(a, b);
    }

    /**
     * Computes the average of two values.
     * @param a first value
     * @param b second value
     */
    public double average(double a, double b) {
        return (a + b) / 2;
    }
}
