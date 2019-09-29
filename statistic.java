import java.util.Scanner;

public class Index {
    static int[] a = new int[1000];
    static int size = 0;

    static void readData() {
        Scanner scanner = new Scanner(System.in);
        while (scanner.hasNextInt()) {
            a[size++] = scanner.nextInt();
        }
    }

    static int[] first_last12(int a[]) {
        int[] result = {0, 0};
        int i = 0;
        boolean flag = false;
        while (i < size) {
            if (a[i] == 12) {
                if (!flag) {
                    result[0] = i + 1;
                    result[1] = i + 1;
                    flag = true;
                } else
                    result[1] = i + 1;
            }
            i++;
        }
        return result;
    }

    static float[] min_max_avg(int a[]) {
        int[] result = {a[0], a[0], 0};
        int i = 0;
        while (i < size) {
            result[0] = Math.min(result[0],a[i]);
            result[1] = Math.max(result[0],a[i]);
            result[2] +=a[i];
            i++;
        }

        return new float[]{result[0],result[1],result[2]/size};
    }

    public static void main(String[] args){
        System.out.println("read data");
        readData();
        System.out.println("find first last 12");
        int[] result1 = first_last12(a);
        System.out.println("first 12:"+result1[0]);
        System.out.println("last 12:"+result1[1]);
        System.out.println("find min max avg");
        float[] result2 = min_max_avg(a);
        System.out.println("min:"+result2[0]);
        System.out.println("max:"+result2[1]);
        System.out.println("avg:"+result2[2]);
    }
}
