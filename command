/home/zmc/Models /home/zmc/042801 /home/zmc/OpenCLBA/kernel-original.cl

valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=log -v ./OpenCLBA-Prod /home/tuan/Desktop/Models/ /home/tuan/Desktop/042801/ /home/tuan/Desktop/OpenCLBA-Local/OpenCLBA/kernel-original.cl


valgrind --leak-check=full  --track-origins=yes --log-file=log -v ./OpenCLBA-Prod /home/tuan/Desktop/Models/ /home/tuan/Desktop/042801/ /home/tuan/Desktop/OpenCLBA-Local/OpenCLBA/kernel-original.cl

1,606 bytes in 60 blocks
==5656==    indirectly lost: 24,948 bytes in 305 blocks
==5656==      possibly lost: 18,082 bytes in 199 blocks
==5656==    still reachable: 694,340 bytes in 2,244 blocks
