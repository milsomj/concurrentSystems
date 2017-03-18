package test;

public class Help {

	//edit K to be 1,3,5 or 7	(although there might not be much point in doing 1)
	private final static int K = 5;
	public static void main(String[] args) {
		System.out.print("sum += ");
		for(int x = 0; x < K; x++){
			for(int y = 0; y < K; y++){
				if(x==K-1&&y==K-1){
					System.out.print("(image[w+"+x+"][h+"+y+"][c] * kernels[m][c]["+x+"]["+y+"]);");
				}
				else{
					System.out.print("(image[w+"+x+"][h+"+y+"][c] * kernels[m][c]["+x+"]["+y+"]) + ");
				}
			}
		}
	}

}
