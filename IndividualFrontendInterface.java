import java.io.IOException;
import java.util.Scanner;

public interface IndividualFrontendInterface{

    /**
     * Expected constructor in the implementing class.
     * Frontend will get backend class and get scanner to read the user's input
     * @param BackendInterface backend class 
     * @param Scanner scanner to read user input
     */
    // public IndividualFrontendInterface(BackendInterface backend, Scanner scanner)

    /**
     *  Starts the main command loop for the user interface.
     *  This should continuously prompt the user until the exit command is received.
     */
    public void runCommandLoop();

    /**
     *  Show statistics about the dataset that includes the number of participants(nodes),
     *  the number of edges(friendships), and the average number of friends.
     */
    public void showStats();

    /**
     *  Prompts the user for two participant, then lists the closest connection
     *  (using Dijkstra's algo I think?)
     */
    public void showClosestConnection();

    /**
     *  Exit the app
     */
    public void exitApp();
}