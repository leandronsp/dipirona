.PHONY: test fmt vet build bench

test:
	go test ./...

fmt:
	go fmt ./...

vet:
	go vet ./...

build:
	go build -o dipirona .

bench:
	go test ./pkg/mat -bench=. -benchtime=1s
