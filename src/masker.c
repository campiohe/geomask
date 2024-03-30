#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
    double lat, lon;
} geo_point;

unsigned int read_geo_point_file(const char* file_path, geo_point** points)
{
    printf("%s\n", file_path);
    FILE *fp = fopen(file_path, "r");

    if (fp == NULL)
    {
        printf("file not found");
        exit(EXIT_FAILURE);
    }

    size_t n_points = 0;
    double lat, lon;
    while(fscanf(fp, "%lf,%lf", &lat, &lon) == 2)
    {
        n_points++;    
    }

    fseek(fp, 0, SEEK_SET);


    printf("%zu\n", n_points);

    *points = (geo_point*)malloc(n_points * sizeof(geo_point));
    if (*points == NULL)
    {
        printf("no memory available\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n_points; i++)
    {
        if (fscanf(fp, "%lf,%lf", &lat, &lon) != 2)
        {
            printf("error reading the file\n");
            exit(EXIT_FAILURE);
        }

        (*points)[i].lat = lat;
        (*points)[i].lon = lon;
    }

    fclose(fp);

    return n_points;
}


unsigned short is_point_inside_polygon(const geo_point* poly, size_t poly_size, const geo_point* p)
{
    // https://observablehq.com/@hg42/untitled
    unsigned short is_inside = 0;

    for (size_t i = 0, j = poly_size - 1; i < poly_size; j = i++)
    {
        unsigned short intersect = (poly[i].lon > p->lon) != (poly[j].lon > p->lon) 
                                && (p->lat < (poly[j].lat - poly[i].lat) * (p->lon - poly[i].lon) / (poly[j].lon - poly[i].lon) + poly[i].lat);
        if (intersect)
        {
            is_inside = !is_inside;
        }
    }

    return is_inside;
}

void points_inside_polygon(const geo_point* mesh, size_t mesh_size, const geo_point* poly, size_t poly_size)
{
    size_t n_points = 0;
    clock_t begin = clock();
    for (size_t i = 0; i < mesh_size; i++)
    {
        if (is_point_inside_polygon(poly, poly_size, &mesh[i]))
        {
            n_points++;
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("n_points: %zu\t[%lf]\n", n_points, time_spent);
}

int main()
{
    geo_point* mesh = NULL;
    unsigned int mesh_size = read_geo_point_file("./data/mesh.bln", &mesh);

    for(int i = 0; i < 5; i++)
    {
        printf("%lf,%lf\n", mesh[i].lat, mesh[i].lon);
    }

    geo_point* poly = NULL;
    unsigned int poly_size = read_geo_point_file("./data/shape.bln", &poly);
    for(int i = 0; i < 5; i++)
    {
        printf("%lf,%lf\n", poly[i].lat, poly[i].lon);
    }

    points_inside_polygon(mesh, mesh_size, poly, poly_size);

    if (mesh != NULL)
        free(mesh);
    
    if (poly != NULL)
        free(poly);

    return 0;
}