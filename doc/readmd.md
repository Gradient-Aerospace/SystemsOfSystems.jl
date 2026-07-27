Make the docs, from this directory:

```
julia --project=. make.jl
```

View the docs:

```
julia --project=. -e 'using LiveServer; serve(; dir = "build", launch_browser = true)'
```
