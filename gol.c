#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

void makePrivateMatrix(int *reciverCounts, int myRank, int cols, bool *reciver, bool *before, bool *after, bool *sender, int startFrom, int arriveTo);

int main(int argc, char *argv[])
{

    int myRank = 0; // Rank del processo

    int numberOfProcesses = 0; // Numero di processi nel COMM_WORLD

    int rows, cols,        //RISPETTIVAMENTE: RIGHE,
        seed, generations; // COLONNE SEME DI GENERAZIONE E GENERAZIONI POSSIBILI DEFINITE DALL'UTENTE

    double start, end; //AVVIO E FINE DEL TIMER

    int surplus = 0; //NUMERO DI RIGHE IN PIU' CHE DEVONO ESSERE, EVENTUALMENTE, ASSEGNATE AI PROCESSORI

    bool *matrix; //MONDO DI GIOCO

    //bool matrix[16] = {0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    int sendTo, reciveFrom; //INTERI CHE DEFINISCONO I PROCESSORI TARGET A CUI E DA CUI INVIARE E RICEVERE LE RIGHE GHOST

    bool *before, *after, *toSendBefore,
        *toSendAfter, *reciver, *sender; //RISPETTIVAMENTE: LINEA GHOST INIZIALE, LINEA GHOST FINALE,
                                         //LINEA GHOST DA INVIARE AL PROCESSORE PRECEDENTE, LINEA GHOST DA INVIARE AL PROCESSORE SUCCESSIVO,
                                         //RECIVER DEL MINI-MONDO DEL PROCESSORE I-ESIMO E MINI-MONDO DI BACKUP PER NON CONTAMINARE RECIVER

    int *reciverCounts, *reciverDisplacement,
        *afterDisplacement, *beforeDisplacement, *beforeAfterCounts; //RISPETTIVAMENTE: ELEMENTI DA RICEVERE,
                                                                     //DA QUALE POSIZIONE PARTIRE PER LA SUDDIVISIONE DEI MINI-MONDI,
                                                                     //DA QUALE ELEMENTO PARTIRE PER LA GENERAZIONE DELLA LINEA GHOST INIZIALE,
                                                                     //DA QUALE ELEMENTO PARTIRE PER LA GENERAZIONE DELLA LINEA GHOST FINALE,
                                                                     //ELEMENTI DA RICEVERE PER LA LINEA GHOST FINALE,

    MPI_Group new_group;
    MPI_Group world_group;
    MPI_Comm MPI_NEW_COMM_WORLD;
    MPI_Request request;
    MPI_Status status[2];
    MPI_Request beforeAfterIrecv[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    //CHECK SUL NUMERO DI ARGOMENTI PASSATI DALL'UTENTE
    if (argc != 5)
    {
        MPI_Finalize();
        exit(0);
    }
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    seed = atoi(argv[3]);
    generations = atoi(argv[4]);

    //CREAZIONE DI UN NUOVO COMUNICATORE PER EVITARE L'INIZIALIZZAZIONE DI PROCESSI IN ECCESSO DA PARTE DELL'UTENTE
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    if (numberOfProcesses > rows)
    {
        int ranges[][3] = {rows, numberOfProcesses - 1, 1};
        MPI_Group_range_excl(world_group, 1, ranges, &new_group);
        MPI_Comm_create(MPI_COMM_WORLD, new_group, &MPI_NEW_COMM_WORLD);
    }
    else
    {
        MPI_Comm_create(MPI_COMM_WORLD, world_group, &MPI_NEW_COMM_WORLD);
    }
    if (MPI_NEW_COMM_WORLD == MPI_COMM_NULL)
    {
        // Bye bye cruel world
        MPI_Finalize();
        exit(0);
    }

    MPI_Comm_rank(MPI_NEW_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_NEW_COMM_WORLD, &numberOfProcesses);
    surplus = rows % numberOfProcesses;

    before = (bool *)calloc(cols, sizeof(bool));
    after = (bool *)calloc(cols, sizeof(bool));
    toSendBefore = (bool *)calloc(cols, sizeof(bool));
    toSendAfter = (bool *)calloc(cols, sizeof(bool));

    reciverCounts = (int *)calloc(numberOfProcesses, sizeof(int));
    beforeAfterCounts = (int *)calloc(numberOfProcesses, sizeof(int));

    reciverDisplacement = (int *)calloc(numberOfProcesses, sizeof(int));
    beforeDisplacement = (int *)calloc(numberOfProcesses, sizeof(int));
    afterDisplacement = (int *)calloc(numberOfProcesses, sizeof(int));

    //DEFINIZIONE DEL NUMERO DI ELEMENTI DA DISTRIBUIRE AI PROCESSORI E DA QUALE ELEMENTO PARTIRE ALL'INTERNO DELLA "MATRICE"
    for (int i = 0; i < numberOfProcesses; i++)
    {
        reciverCounts[i] = (surplus - i > 0) ? ((cols * (rows / numberOfProcesses)) + cols) : (cols * (rows / numberOfProcesses));
        beforeAfterCounts[i] = cols;
        reciverDisplacement[i] = (i == 0) ? 0 : (reciverCounts[i - 1] + reciverDisplacement[i - 1]);
        beforeDisplacement[i] = (i == 0) ? ((rows * cols) - cols) : reciverDisplacement[i] - cols;
        afterDisplacement[i] = (i == numberOfProcesses - 1) ? 0 : reciverCounts[i] + reciverDisplacement[i];
    }

    reciver = (bool *)calloc(reciverCounts[myRank], sizeof(bool));
    sender = (bool *)calloc(reciverCounts[myRank], sizeof(bool));

    //INIZIALIZZAZIONE E STAMPA DEL MONDO DI GIOCO
    if (myRank == 0)
    {
        srand(seed);
        matrix = (bool *)calloc((rows * cols), sizeof(bool));
        printf("Seed: %d\nGeneration 0 auto-generated:\n", seed);
        fflush(stdout);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i * cols + j] = rand() % 2;
                printf("%d\t", matrix[i * cols + j]);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_NEW_COMM_WORLD);

    //AVVIO DEL TIMER
    start = MPI_Wtime();

    //DISTRIBUZIONE DEL MONDO DI GIOCO
    MPI_Scatterv(matrix, reciverCounts, reciverDisplacement, MPI_C_BOOL, reciver, reciverCounts[myRank], MPI_C_BOOL, 0, MPI_NEW_COMM_WORLD);

    for (int rounds = 0; rounds < generations; rounds++)
    {
        //PREPARAZIONE DELLA GHOST LINE PER IL PROCESSO PRECEDENTE
        for (int i = 0; i < cols; i++)
        {
            toSendAfter[i] = reciver[i];
        }

        sendTo = (myRank == 0) ? (numberOfProcesses - 1) : (myRank - 1);
        MPI_Isend(toSendAfter, cols, MPI_C_BOOL, sendTo, 0, MPI_NEW_COMM_WORLD, &request);

        reciveFrom = (myRank == numberOfProcesses - 1) ? 0 : (myRank + 1);
        MPI_Irecv(after, cols, MPI_C_BOOL, reciveFrom, 0, MPI_NEW_COMM_WORLD, &beforeAfterIrecv[0]);

        //PREPARAZIONE DELLA GHOST LINE PER IL PROCESSO SUCCESSIVO
        if (reciverCounts[myRank] / cols == 1)
        {
            for (int i = 0; i < cols; i++)
            {
                toSendBefore[i] = reciver[i];
            }
            sendTo = (myRank == numberOfProcesses - 1) ? 0 : myRank + 1;
            MPI_Isend(toSendBefore, cols, MPI_C_BOOL, sendTo, 0, MPI_NEW_COMM_WORLD, &request);
            reciveFrom = (myRank == 0) ? numberOfProcesses - 1 : myRank - 1;
            MPI_Irecv(before, cols, MPI_C_BOOL, reciveFrom, 0, MPI_NEW_COMM_WORLD, &beforeAfterIrecv[1]);
        }
        else
        {
            int j = 0;
            for (int i = (reciverCounts[myRank] - cols); i < reciverCounts[myRank]; i++)
            {
                toSendBefore[j] = reciver[i];
                j++;
            }
            sendTo = (myRank == numberOfProcesses - 1) ? 0 : myRank + 1;
            MPI_Isend(toSendBefore, cols, MPI_C_BOOL, sendTo, 0, MPI_NEW_COMM_WORLD, &request);
            reciveFrom = (myRank == 0) ? numberOfProcesses - 1 : myRank - 1;
            MPI_Irecv(before, cols, MPI_C_BOOL, reciveFrom, 0, MPI_NEW_COMM_WORLD, &beforeAfterIrecv[1]);
        }

        // SE LE RIGHE RICEVUTE SONO MINORI DI DUE
        if (reciverCounts[myRank] / cols <= 2)
        {
            //ATTENDO LE GHOST LINES
            MPI_Waitall(2, beforeAfterIrecv, status);

            //CALCOLO IL NUOVO MINI-MONDO DI GIOCO
            makePrivateMatrix(reciverCounts, myRank, cols, reciver, before, after, sender, 0, reciverCounts[myRank]);
        }

        // SE LE RIGHE RICEVUTE SONO MAGGIORI DI DUE
        else
        {
            //CALCOLO IL NUOVO MINI-MONDO DI GIOCO ESCLUDENDO LA PRIMA ED ULTIMA LINEA
            makePrivateMatrix(reciverCounts, myRank, cols, reciver, before, after, sender, 1, (reciverCounts[myRank] / cols) - 1);

            //ATTENDO LE GHOST LINES
            MPI_Waitall(2, beforeAfterIrecv, status);

            //PRIMA LINEQ
            makePrivateMatrix(reciverCounts, myRank, cols, reciver, before, after, sender, 0, 1);

            //ULTIMA RIGA
            makePrivateMatrix(reciverCounts, myRank, cols, reciver, before, after, sender, (reciverCounts[myRank] / cols) - 1, reciverCounts[myRank] / cols);
        }

        //SENDER POSSIEDE IL NUOVO MONDO DI GIOCO
        memcpy(reciver, sender, reciverCounts[myRank]);

        //SINCRONIZZO I PROCESSORI
        MPI_Barrier(MPI_NEW_COMM_WORLD);
    }
    //ASSEMBLO I MINI-MONDI DI GIOCO
    MPI_Gatherv(reciver, reciverCounts[myRank], MPI_C_BOOL, matrix, reciverCounts, reciverDisplacement, MPI_C_BOOL, 0, MPI_NEW_COMM_WORLD);

    //STAMPO IL NUOVO MONDO DI GIOCO
    if (myRank == 0)
    {
        printf("Generation %d:\n", generations + 1);
        fflush(stdout);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                printf("%d\t", matrix[i * cols + j]);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }
    if (myRank == 0)
    {
        free(matrix);
    }
    free(before);
    free(after);
    free(reciver);
    free(sender);
    free(reciverCounts);
    free(reciverDisplacement);
    free(afterDisplacement);
    free(beforeDisplacement);
    free(beforeAfterCounts);
    end = MPI_Wtime();
    MPI_Finalize();
    if (myRank == 0)
    {
        printf("Time in ms = %f\n", end - start);
    }
}

void makePrivateMatrix(int *reciverCounts, int myRank, int cols, bool *reciver, bool *before, bool *after, bool *sender, int startFrom, int arriveTo)
{
    //STATO DELLA CELLA E DELLE CELLE VICINE
    int block[9];

    // # DELLE CELLE VICINE VIVE
    int livingCells = 0;
    for (int i = startFrom; i < arriveTo / cols; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            //focus = 0, left = 1, right = 2, up = 3, down = 4, lUpCorner = 5, lDownCorner = 6, rUpCorner = 7, rDownCorner = 8;
            block[0] = reciver[i * cols + j];
            block[1] = (j == 0) ? (reciver[((i * cols + cols) - 1)]) : (reciver[((i * cols + j) - 1)]);
            block[2] = (j == cols - 1) ? (reciver[(i * cols)]) : (reciver[((i * cols + j) + 1)]);
            block[3] = (i == 0) ? (before[j]) : (reciver[(((i - 1) * cols) + j)]);
            block[4] = (i == ((reciverCounts[myRank] / cols) - 1)) ? (after[j]) : (reciver[(((i + 1) * cols) + j)]);

            if (i == 0)
            {
                block[5] = (j == 0) ? before[cols - 1] : before[j - 1];
            }
            else if (i > 0)
            {
                block[5] = (j == 0) ? reciver[((((i - 1) * cols) + cols) - 1)] : reciver[((((i - 1) * cols) + j) - 1)];
            }

            if (i == ((reciverCounts[myRank] / cols) - 1))
            {
                block[6] = (j == 0) ? after[cols - 1] : after[j - 1];
            }
            else if (i < ((reciverCounts[myRank] / cols) - 1))
            {
                block[6] = (j == 0) ? reciver[((((i + 1) * cols) + cols) - 1)] : reciver[((((i + 1) * cols) + j) - 1)];
            }

            if (i == 0)
            {
                block[7] = (j == cols - 1) ? before[0] : before[j + 1];
            }
            else if (i > 0)
            {
                block[7] = (j == cols - 1) ? reciver[((i - 1) * cols)] : reciver[((((i - 1) * cols) + j) + 1)];
            }

            if (i == ((reciverCounts[myRank] / cols) - 1))
            {
                block[8] = (j == cols - 1) ? after[0] : after[j + 1];
            }
            else if (i < ((reciverCounts[myRank] / cols) - 1))
            {
                block[8] = (j == cols - 1) ? reciver[(i + 1) * cols] : reciver[((((i + 1) * cols) + j) + 1)];
            }

            //ANALIZZO IL NUMERO DI CELLE VIVE ED APPLICO LE REGOLE DI VIVO-MORTO
            for (int k = 1; k < 9; k++)
            {
                livingCells = livingCells + block[k];
                if (block[0] == 1 && livingCells > 3)
                {
                    sender[i * cols + j] = 0;
                    break;
                }
            }

            if (block[0] == 0 && livingCells == 3)
            {
                sender[i * cols + j] = 1;
            }
            else if (block[0] == 1 && (livingCells == 2 || livingCells == 3))
            {
                sender[i * cols + j] = 1;
            }
            else if (block[0] == 1 && livingCells < 2)
            {
                sender[i * cols + j] = 0;
            }
            livingCells = 0;
        }
    }
}