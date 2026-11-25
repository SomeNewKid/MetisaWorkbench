var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () => Results.Content("<h1>Home of Metisa</h1>", "text/html; charset=utf-8"));
app.MapGet("/hello", () => Results.Content("<h1>Hello from Metisa</h1>", "text/html; charset=utf-8"));

app.Run();

/*
dotnet publish .\src\metisa_sandbox\http_server_sidecar\WebApp\WebApp.csproj `
    --configuration Release `
    --output .\src\metisa_sandbox\http_server_sidecar\publish
*/